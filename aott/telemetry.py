"""
Grab AO telemetry and science frames from the dao shared memories and write one
observation HDF5 file, with the same layout as simulation/DataGeneration.ipynb:

    python -m aott.telemetry <target> <duration_s> [--no-simbad] [--controlled-modes N]

The threads, the shared memories each one records and the ones read once, and where
each of them goes in the file, come from config/data_grabber.toml; the instrument
values from config/instrument.toml (see aott/config.py). The file is written to
<hdf5_dir>/<UTC date>/<target>_<UTC date>T<HH-MM-SS>.hdf5, time of the start of the
acquisition. The analysis and the report are run separately
(python -m aott.AutomaticAnalysis), or together with the grab by python -m aott.observe.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

import dao
import h5py
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroquery.simbad import Simbad

from aott.config import LoadInstrument
from aott.DataGrabber import AsList, LoadConfig, ReadSample, RecordInParallel, Stream, StreamConfig, Window


# Calibration matrices stored with the DM actuators along their first axis
ACTUATOR_MATRICES = ("Calibration/M2C", "Calibration/Z2C")


def QueryTarget(name):
    """
    SIMBAD entry of `name` as a dict: main_id, coord (ICRS SkyCoord) and the V, R, J, H
    magnitudes (NaN when SIMBAD has none). None if SIMBAD doesn't know the target.
    """
    Simbad.add_votable_fields("V", "R", "J", "H")
    result = Simbad.query_object(name)
    if result is None or len(result) == 0:
        return None

    row = result[0]
    star = {
        "main_id": str(row["main_id"]),
        "coord": SkyCoord(ra=row["ra"], dec=row["dec"], unit=(u.deg, u.deg), frame="icrs"),
    }
    for band in ("V", "R", "J", "H"):
        star[band] = np.nan if np.ma.is_masked(row[band]) else float(row[band])
    return star


def TargetAltAz(coord, unix_time, site):
    """Altitude/azimuth of `coord` at `unix_time`, seen from the [site] of config/instrument.toml."""
    location = EarthLocation(lat=site["latitude_deg"] * u.deg,
                             lon=site["longitude_deg"] * u.deg,
                             height=site["height_m"] * u.m)
    frame = AltAz(obstime=Time(unix_time, format="unix"), location=location)
    return coord.transform_to(frame)


def LoadArray(path, window=None, sliceable=True):
    """
    Read an array from a .npy file or a dao .im.shm file, optionally cropped to `window`.
    A `sliceable` shm crops it itself, otherwise the whole image is read and cropped here.
    """
    window = window or {}
    if str(path).endswith(".shm"):
        if sliceable:
            return np.asarray(dao.shm(path).get_data(**window)).squeeze()
        data = np.asarray(dao.shm(path).get_data()).squeeze()
    else:
        data = np.load(path).squeeze()
    if window:
        data = data[window["y"], window["x"]]
    return data


def ActuatorsFirst(matrix, n_act, name):
    """`matrix` as (n_act, n), transposed if it came as (n, n_act). If ambiguous, assumed correct"""
    matrix = np.asarray(matrix).squeeze()
    if matrix.ndim == 2:# and matrix.shape[0] != matrix.shape[1]:
        if matrix.shape[0] == n_act:
            return matrix
        if matrix.shape[1] == n_act:
            return matrix.T
    sys.exit(f"{name}: can't tell which axis holds the {n_act} DM actuators in shape {matrix.shape}")


def ReadStatic(config, entry):
    """The array of a static entry: its `value`, or its `source` read once, cropped
    like the stream its `window_of` names."""
    if "value" in entry:
        return np.asarray(entry["value"])
    stream = StreamConfig(config, entry["window_of"]) if "window_of" in entry else {}
    return LoadArray(entry["source"], Window(stream), stream.get("sliceable", True))


def Scalar(array):
    """The [0, 0] element of a scalar setting's image."""
    return np.asarray(array).flat[0]


def WriteAttr(file, destination, value):
    """Write `value` to the attr destination "path@Name" ("@Name" for the root), creating
    `path` as a group if nothing is there yet."""
    path, _, name = destination.rpartition("@")
    if not path:
        target = file
    elif path in file:
        target = file[path]
    else:
        target = file.require_group(path)
    target.attrs[name] = value


def acquire(target, duration, no_simbad=False, controlled_modes=None):
    """
    Grab `duration` seconds of every thread of the data grabber config for `target`,
    write the observation HDF5 file and return its path. Exits before grabbing if the
    config is incomplete or, unless `no_simbad`, SIMBAD doesn't know the target.
    `controlled_modes` is the number of modes the loop corrects, saved as
    Calibration.attrs["Total_Number_Of_Controlled_Modes"]; by default, the
    number of columns of M2C.
    """
    config = LoadConfig()
    instrument = LoadInstrument()
    camera = config["acquisition"]["analysed_camera"]
    if camera not in instrument["science_camera"]:
        sys.exit(f"analysed_camera = '{camera}' has no [science_camera.{camera}] section in the instrument config")

    # Query SIMBAD before grabbing, so a typo in the target name is caught before any data is taken
    star = None
    if not no_simbad:
        try:
            star = QueryTarget(target)
        except Exception as e:
            sys.exit(f"SIMBAD query failed ({e}). Pass --no-simbad to grab anyway.")
        if star is None:
            sys.exit(f"SIMBAD doesn't know '{target}'. Check the name, or pass --no-simbad to grab anyway.")
        print(f"{target}: SIMBAD {star['main_id']}, V = {star['V']}")

    # Streams per thread, its pacer first, in the order their data comes back
    statics = config.get("static", {})
    threads = {}
    stream_configs = {}
    n_act = None
    for thread_name, thread in config["threads"].items():
        names = [thread["pacer"]] + [name for name in thread["streams"] if name != thread["pacer"]]
        stream_configs[thread_name] = [thread["streams"][name] for name in names]
        streams = []
        for stream_config in stream_configs[thread_name]:
            stream = Stream(dao.shm(stream_config["shm"]),
                            keep_every=stream_config.get("keep_every", 1),
                            window=Window(stream_config),
                            sliceable=stream_config.get("sliceable", True),
                            record_timestamps="timestamps" in stream_config,
                            mean="mean" in stream_config,
                            saturation_level=stream_config.get("saturation_level"))
            if stream_config["dataset"] == "WFS/DM_commands":
                n_act = np.asarray(stream.shm.get_data()).size
            streams.append(stream)
        threads[thread_name] = [streams, thread.get("semaphore", config["acquisition"]["semaphore"]), None]

    # Static values, read once. The calibration arrays are checked against the DM
    # command size here, so a wrong file stops the script before the acquisition.
    if n_act != instrument["dm"]["n_actuators"]:
        print(f"WARNING: the DM commands have {n_act} actuators, the instrument file says "
              f"{instrument['dm']['n_actuators']} (saved as Total_Number_Of_Actuators)")
    static = {}
    for name, entry in statics.items():
        static[name] = ReadStatic(config, entry)
        if any(path in ACTUATOR_MATRICES for path in AsList(entry.get("dataset", []))):
            static[name] = ActuatorsFirst(static[name], n_act, f"static.{name}")

    for thread_name, thread in config["threads"].items():
        streams = threads[thread_name][0]
        if "rate" in thread:
            threads[thread_name][2] = float(Scalar(static[thread["rate"]]))
        # Backgrounds are cropped like their stream (window_of), and subtracted from every sample
        for k, stream_config in enumerate(stream_configs[thread_name]):
            if "background" in stream_config:
                background = static[stream_config["background"]].astype(np.float32)
                frame_shape = np.squeeze(ReadSample(streams[k])).shape
                if background.shape != frame_shape:
                    sys.exit(f"static.{stream_config['background']}: shape {background.shape}, "
                             f"the frames of threads.{thread_name} are {frame_shape}")
                streams[k] = streams[k]._replace(background=background)

    n_modes = next(static[name].shape[1] for name, entry in statics.items()
                   if "Calibration/M2C" in AsList(entry.get("dataset", [])))
    if controlled_modes is None:
        controlled_modes = n_modes
    elif not 1 <= controlled_modes <= n_modes:
        sys.exit(f"{controlled_modes} controlled modes: M2C has {n_modes} modes")

    recordings = RecordInParallel({name: tuple(thread) for name, thread in threads.items()}, duration)

    for thread_name, recording in recordings.items():
        rate = threads[thread_name][2]
        expected = f", about {rate * duration:.0f} expected" if rate else ""
        print(f"{thread_name}: {len(recording.timestamps)} iterations{expected}")

    start_time = min(recording.timestamps[0] for recording in recordings.values())
    start = datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc)

    if star is not None:
        altaz = TargetAltAz(star["coord"], start_time, instrument["site"])
        elevation = altaz.alt.deg
        print(f"Elevation at start: {elevation:.2f} deg, azimuth: {altaz.az.deg:.2f} deg")
    else:
        elevation = np.nan

    save_folder = Path(config["output"]["hdf5_dir"]) / f"{start:%Y-%m-%d}"
    save_folder.mkdir(parents=True, exist_ok=True)
    safe_target = re.sub(r"[^A-Za-z0-9+.-]+", "_", target).strip("_")
    hdf5_path = save_folder / f"{safe_target}_{start:%Y-%m-%dT%H-%M-%S}.hdf5"

    with h5py.File(hdf5_path, "w-") as file:
        file.attrs["Instrument"] = instrument["instrument"]["name"]
        file.attrs["Telescope"] = instrument["instrument"]["telescope"]

        for thread_name, recording in recordings.items():
            for stream_config, data, timestamps, mean, peak in zip(
                    stream_configs[thread_name], recording.data, recording.stream_timestamps,
                    recording.means, recording.maxima):
                dset = file.create_dataset(stream_config["dataset"], data=data)
                dset.attrs["Frame_Step"] = stream_config.get("keep_every", 1)
                if "description" in stream_config:
                    dset.attrs["Description"] = stream_config["description"]
                if "window" in stream_config:
                    # [y start, y stop, x start, x stop] of the frame, in pixels
                    dset.attrs["Window"] = [*stream_config["window"]["y"], *stream_config["window"]["x"]]
                if "timestamps" in stream_config:
                    file.create_dataset(stream_config["timestamps"], data=timestamps)
                if "mean" in stream_config:
                    file.create_dataset(stream_config["mean"], data=mean)
                if peak is not None:
                    level = stream_config["saturation_level"]
                    dset.attrs["Max_Value"] = peak
                    dset.attrs["Saturation_Level"] = level
                    dset.attrs["Saturated"] = bool(peak > level)
                    if peak > level:
                        print(f"WARNING: {stream_config['dataset']} reached {peak}, above {level}")

        # Datasets before attrs, so an attr destination on a dataset finds it
        for name, entry in statics.items():
            for path in AsList(entry.get("dataset", [])):
                dset = file.create_dataset(path, data=static[name])
                if "description" in entry:
                    dset.attrs["Description"] = entry["description"]
        for name, entry in statics.items():
            for destination in AsList(entry.get("attr", [])):
                WriteAttr(file, destination, Scalar(static[name]))

        grp_science = file.require_group("Science")
        grp_science.attrs["Target"] = target
        # The Science/<camera> group the analysis reads
        grp_science.attrs["Analysed_Camera"] = camera
        # degrees, at the start of the acquisition
        grp_science.attrs["Elevation"] = elevation
        if star is not None:
            grp_science.attrs["SIMBAD_ID"] = star["main_id"]
            # ICRS, degrees
            grp_science.attrs["RA"] = star["coord"].ra.deg
            grp_science.attrs["Dec"] = star["coord"].dec.deg
            grp_science.attrs["Azimuth"] = altaz.az.deg
            grp_science.attrs["Vmag"] = star["V"]
            grp_science.attrs["Rmag"] = star["R"]
            grp_science.attrs["Jmag"] = star["J"]
            grp_science.attrs["Hmag"] = star["H"]

        for name, science_camera in instrument["science_camera"].items():
            if f"Science/{name}/Science_PSFs" not in file:
                continue
            dset_science = file[f"Science/{name}/Science_PSFs"]
            dset_science.attrs["Sampling"] = science_camera["sampling_at_calibration"]
            dset_science.attrs["Wavelength"] = science_camera["wvl_nm"] * 1e-9
            dset_science.attrs["Bandpass"] = science_camera["bandpass_nm"] * 1e-9
            dset_science.attrs["Calibration_Wavelength"] = science_camera["calibration_wvl_nm"] * 1e-9

        grp_calibration = file.require_group("Calibration")
        grp_calibration.attrs["Diameter"] = instrument["telescope"]["diameter_m"]
        grp_calibration.attrs["Obstruction_ratio"] = instrument["telescope"]["obstruction_ratio"]
        grp_calibration.attrs["AO_Calibration_Wavelength"] = instrument["wfs"]["interaction_matrix_wvl_nm"] * 1e-9
        grp_calibration.attrs["SkyCalibPupilRatio"] = instrument["dm"]["sky_calib_pupil_ratio"]
        grp_calibration.attrs["Actuators_in_diameter"] = instrument["dm"]["actuators_in_diameter"]
        grp_calibration.attrs["Total_Number_Of_Actuators"] = instrument["dm"]["n_actuators"]
        grp_calibration.attrs["Total_Number_Of_Controlled_Modes"] = controlled_modes
        grp_calibration.attrs["r0_reference_wvl"] = instrument["conventions"]["r0_reference_wvl_nm"] * 1e-9
        if "Interaction_Matrix" in grp_calibration:
            grp_calibration["Interaction_Matrix"].attrs["Wavelength"] = instrument["wfs"]["interaction_matrix_wvl_nm"] * 1e-9

    print(hdf5_path)
    return hdf5_path


def main():
    parser = argparse.ArgumentParser(description="Grab telemetry and PSFs and write the observation HDF5 file.")
    parser.add_argument("target", type=str, help="name of the target, as SIMBAD knows it")
    parser.add_argument("duration", type=float, help="acquisition time in seconds")
    parser.add_argument("--no-simbad", action="store_true",
                        help="grab without the SIMBAD query: no magnitudes or coordinates, NaN elevation")
    parser.add_argument("--controlled-modes", type=int, default=None,
                        help="number of modes the loop corrects (default: the number of columns of M2C)")
    args = parser.parse_args()
    acquire(args.target, args.duration, args.no_simbad, args.controlled_modes)


if __name__ == "__main__":
    main()
