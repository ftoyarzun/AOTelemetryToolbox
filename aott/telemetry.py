"""
Grab AO telemetry and science frames from the dao shared memories and write one
observation HDF5 file, with the same layout as simulation/DataGeneration.ipynb:

    python -m aott.telemetry <target> <duration_s> [--no-simbad]

Settings come from config/data_grabber.toml and config/instrument.toml (see aott/config.py).
The file is written to <hdf5_dir>/<UTC date>/<target>_<UTC date>T<HH-MM-SS>.hdf5, time
of the start of the acquisition. The analysis and the report are run separately
(python -m aott.AutomaticAnalysis).
"""
import argparse
import datetime
import math
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
from aott.DataGrabber import LoadConfig, Stream, RecordInParallel


# Optional [calibration] keys of the data grabber config, and where they go in the HDF5 file.
# A key that is left out (or "TODO") is not written.
OPTIONAL_ARRAYS = {
    "interaction_matrix": "Calibration/Interaction_Matrix",
    "wfs_dark": "WFS/Dark",
    "wfs_reference_frame": "WFS/Reference_Frame",
    "dm_flat": "WFS/DM_flat",
    "dm_offset": "WFS/DM_offset",
    "science_dark": "Science/Dark",
}


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


def LoadArray(path, window=None):
    """Read an array from a .npy file or a dao .im.shm file, optionally cropped to `window`."""
    window = window or {}
    if str(path).endswith(".shm"):
        return np.asarray(dao.shm(path).get_data(**window)).squeeze()
    data = np.load(path).squeeze()
    if window:
        data = data[window["y"], window["x"]]
    return data


def ActuatorsFirst(matrix, n_act, name):
    """`matrix` as (n_act, n), transposed if it came as (n, n_act). Exits if the orientation is ambiguous."""
    matrix = np.asarray(matrix).squeeze()
    if matrix.ndim == 2 and matrix.shape[0] != matrix.shape[1]:
        if matrix.shape[0] == n_act:
            return matrix
        if matrix.shape[1] == n_act:
            return matrix.T
    sys.exit(f"{name}: can't tell which axis holds the {n_act} DM actuators in shape {matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Grab telemetry and PSFs and write the observation HDF5 file.")
    parser.add_argument("target", type=str, help="name of the target, as SIMBAD knows it")
    parser.add_argument("duration", type=float, help="acquisition time in seconds")
    parser.add_argument("--no-simbad", action="store_true",
                        help="grab without the SIMBAD query: no magnitudes or coordinates, NaN elevation")
    args = parser.parse_args()

    config = LoadConfig()
    instrument = LoadInstrument()

    # Query SIMBAD before grabbing, so a typo in the target name is caught before any data is taken
    star = None
    if not args.no_simbad:
        try:
            star = QueryTarget(args.target)
        except Exception as e:
            sys.exit(f"SIMBAD query failed ({e}). Pass --no-simbad to grab anyway.")
        if star is None:
            sys.exit(f"SIMBAD doesn't know '{args.target}'. Check the name, or pass --no-simbad to grab anyway.")
        print(f"{args.target}: SIMBAD {star['main_id']}, V = {star['V']}")

    shm_paths = config["shm"]
    wfs_frames_shm = dao.shm(shm_paths["wfs"]["frames"])
    wfs_measurements_shm = dao.shm(shm_paths["wfs"]["measurements"])
    dm_shm = dao.shm(shm_paths["dm"]["commands"])
    loop_cmd_shm = dao.shm(shm_paths["loop"]["cmd"])
    psf_shm = dao.shm(shm_paths["science"]["frames"])

    sem_nb = config["acquisition"]["semaphore"]
    wfs_frame_step = config["acquisition"].get("wfs_frame_step", 1)

    # Optional window of the science frame to read, empty means the whole frame
    crop = config["acquisition"].get("science_crop")
    psf_window = {"y": slice(*crop["y"]), "x": slice(*crop["x"])} if crop else {}

    # Static values, read once. The calibration arrays are checked against the DM
    # command size here, so a wrong file stops the script before the acquisition.
    wfs_pup = dao.shm(shm_paths["wfs"]["valid_pixel_map"]).get_data()
    loop_gain = dao.shm(shm_paths["loop"]["gain"]).get_data()[0, 0]
    loop_leak = dao.shm(shm_paths["loop"]["leak"]).get_data()[0, 0]
    wfs_fps = dao.shm(shm_paths["wfs"]["fps"]).get_data()[0, 0]
    wfs_gain = dao.shm(shm_paths["wfs"]["gain"]).get_data()[0, 0]
    sci_dit = dao.shm(shm_paths["science"]["dit"]).get_data()[0, 0]
    sci_fps = dao.shm(shm_paths["science"]["fps"]).get_data()[0, 0]
    sci_gain = dao.shm(shm_paths["science"]["gain"]).get_data()[0, 0]

    n_act = dm_shm.get_data().size
    if n_act != instrument["dm"]["n_actuators"]:
        print(f"WARNING: the DM commands have {n_act} actuators, the instrument file says "
              f"{instrument['dm']['n_actuators']} (saved as Total_Number_Of_Actuators)")

    m2c = ActuatorsFirst(dao.shm(shm_paths["dm"]["m2c"]).get_data(), n_act, "shm.dm.m2c")
    z2c = ActuatorsFirst(LoadArray(config["calibration"]["Z2C"]), n_act, "calibration.Z2C")

    optional = {}
    for key, path in config["calibration"].items():
        if key in OPTIONAL_ARRAYS and path != "TODO":
            optional[key] = LoadArray(path, psf_window if key == "science_dark" else None)

    wfs_rec, psf_rec = RecordInParallel(
        [
            [Stream(wfs_frames_shm, keep_every=wfs_frame_step), Stream(dm_shm), Stream(wfs_measurements_shm),
             Stream(loop_cmd_shm)],
            [Stream(psf_shm, window=psf_window)],
        ],
        args.duration,
        sem_nb,
    )

    print(f"WFS Camera: {len(wfs_rec.timestamps)} frames")
    print(f"PSF Camera: {len(psf_rec.timestamps)} frames")

    wfs_frames, dm_commands, wfs_measurements, loop_status = wfs_rec.data
    (psf_frames,) = psf_rec.data

    start_time = min(wfs_rec.timestamps[0], psf_rec.timestamps[0])
    start = datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc)

    if star is not None:
        altaz = TargetAltAz(star["coord"], start_time, instrument["site"])
        elevation = altaz.alt.deg
        print(f"Elevation at start: {elevation:.2f} deg, azimuth: {altaz.az.deg:.2f} deg")
    else:
        elevation = np.nan

    save_folder = Path(config["output"]["hdf5_dir"]) / f"{start:%Y-%m-%d}"
    save_folder.mkdir(parents=True, exist_ok=True)
    safe_target = re.sub(r"[^A-Za-z0-9+.-]+", "_", args.target).strip("_")
    hdf5_path = save_folder / f"{safe_target}_{start:%Y-%m-%dT%H-%M-%S}.hdf5"

    with h5py.File(hdf5_path, "w-") as file:
        file.attrs["Instrument"] = instrument["instrument"]["name"]
        file.attrs["Telescope"] = instrument["instrument"]["telescope"]

        grp_wfs = file.create_group("WFS")
        grp_wfs.attrs["Loop_Gain"] = loop_gain
        grp_wfs.attrs["Loop_Leak"] = loop_leak
        grp_wfs.attrs["Loop_Freq"] = wfs_fps

        dset_wfs = grp_wfs.create_dataset("WFS_Images", data=wfs_frames)
        dset_wfs.attrs["Gain"] = wfs_gain
        dset_wfs.attrs["FPS"] = wfs_fps
        dset_wfs.attrs["Frame_Step"] = wfs_frame_step
        grp_wfs.create_dataset("WFS_TimeStamps", data=wfs_rec.timestamps[::wfs_frame_step])

        grp_wfs.create_dataset("Valid_Pixel_Map", data=wfs_pup)
        grp_wfs.create_dataset("DM_commands", data=dm_commands)
        grp_wfs.create_dataset("DM_TimeStamps", data=wfs_rec.timestamps)
        grp_wfs.create_dataset("WFS_measurements", data=wfs_measurements)
        # Loop command at every loop iteration. The notebook stores it as the loop_status attribute,
        # a dataset here because attributes are limited to 64 kB. The analysis derives the
        # open/closed status from the DM commands and doesn't read it.
        grp_wfs.create_dataset("loop_status", data=loop_status)

        grp_science = file.create_group("Science")
        grp_science.attrs["Target"] = args.target
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
        grp_science.create_dataset("PSF_TimeStamps", data=psf_rec.timestamps)

        science_camera = instrument["science_camera"]
        dset_science = grp_science.create_dataset("Science_PSFs", data=psf_frames)
        dset_science.attrs["Exposure_Time"] = sci_dit
        dset_science.attrs["FPS"] = sci_fps
        dset_science.attrs["Gain"] = sci_gain
        dset_science.attrs["Sampling"] = science_camera["sampling_at_calibration"]
        dset_science.attrs["Wavelength"] = science_camera["wvl_nm"] * 1e-9
        dset_science.attrs["Bandpass"] = science_camera["bandpass_nm"] * 1e-9

        grp_calibration = file.create_group("Calibration")
        grp_calibration.create_dataset("M2C", data=m2c)
        grp_calibration.create_dataset("Z2C", data=z2c)
        grp_calibration.attrs["Diameter"] = instrument["telescope"]["diameter_m"]
        grp_calibration.attrs["Obstruction_ratio"] = instrument["telescope"]["obstruction_ratio"]
        grp_calibration.attrs["Science_Calibration_Wavelength"] = science_camera["calibration_wvl_nm"] * 1e-9
        grp_calibration.attrs["AO_Calibration_Wavelength"] = instrument["wfs"]["interaction_matrix_wvl_nm"] * 1e-9
        grp_calibration.attrs["SkyCalibPupilRatio"] = instrument["dm"]["sky_calib_pupil_ratio"]
        grp_calibration.attrs["Actuators_in_diameter"] = instrument["dm"]["actuators_in_diameter"]
        grp_calibration.attrs["Total_Number_Of_Actuators"] = instrument["dm"]["n_actuators"]
        grp_calibration.attrs["Total_Number_Of_Controlled_Modes"] = m2c.shape[1]
        grp_calibration.attrs["r0_reference_wvl"] = instrument["conventions"]["r0_reference_wvl_nm"] * 1e-9

        for key, data in optional.items():
            file.create_dataset(OPTIONAL_ARRAYS[key], data=data)
        if "interaction_matrix" in optional:
            file["Calibration/Interaction_Matrix"].attrs["Wavelength"] = instrument["wfs"]["interaction_matrix_wvl_nm"] * 1e-9

    print(hdf5_path)


if __name__ == "__main__":
    main()
