"""The data grabber and python -m aott.observe end to end, with a fake dao that replays a
synthetic observation."""
import copy
import sys
import time
import types
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

from conftest import analyse, requires_typst
from synthetic import write_observation

from aott.config import CONFIG_DIR, LoadInstrument
from aott.DataGrabber import ConfigProblems

# How far behind this machine's clock the fake shm timestamps are [s]
SHM_CLOCK_OFFSET = 0.5


def fake_dao(source, sliceable=True, background=0):
    """A dao module whose shared memories replay `source`: the loop at its Loop_Freq, the
    science cameras ("psf", and "psf2" for a second one) at its FPS, with the loop closed.
    The WFS frames are filled with their frame count. The science shms add `background` to
    the frames and, unless `sliceable`, refuse a window. get_timestamp() is
    SHM_CLOCK_OFFSET behind this machine's clock."""
    with h5py.File(source, "r") as f:
        data = dict(dm=f["WFS/DM_commands"][:], meas=f["WFS/WFS_measurements"][:],
                    psf=f["Science/camera/Science_PSFs"][:], pup=f["WFS/Valid_Pixel_Map"][:],
                    dm_map=f["WFS/DM_Map"][:], m2c=f["Calibration/M2C"][:])
        loop_fps = float(f["WFS"].attrs["Loop_Freq"])
        science_fps = float(f["Science/camera/Science_PSFs"].attrs["FPS"])
    scalars = dict(loop_gain=0.5, loop_leak=0.99, wfs_fps=loop_fps, wfs_gain=1, sci_dit=1 / science_fps,
                   sci_fps=science_fps, sci_gain=1, loop_cmd=1)
    count = {"wfs": 0, "psf": 0, "psf2": 0}
    due = {key: time.perf_counter() for key in count}

    def wait(key, period):
        # Busy wait: time.sleep is too coarse for 1 kHz on Windows
        due[key] += period
        while time.perf_counter() < due[key]:
            pass
        count[key] += 1

    class shm:
        def __init__(self, path):
            self.path = path.removesuffix(".im.shm")

        def get_data(self, check=False, semNb=None, **window):
            if self.path == "wfs_frames":
                if check:
                    wait("wfs", 1 / loop_fps)
                return np.full((8, 8), float(count["wfs"]))
            if self.path in ("dm", "meas"):
                return data[self.path][count["wfs"] % len(data[self.path])]
            if self.path in ("psf", "psf2", "psf_background"):
                if window and not sliceable:
                    raise TypeError("this shm can't slice")
                if self.path == "psf_background":
                    frame = np.full(data["psf"].shape[1:], background)
                else:
                    if check:
                        wait(self.path, 1 / science_fps)
                    frame = data["psf"][count[self.path] % len(data["psf"])] + background
                return frame[window["y"], window["x"]] if window else frame
            if self.path in ("pup", "dm_map", "m2c"):
                return data[self.path]
            if self.path == "modulator":
                return np.arange(5.0).reshape(5, 1)
            return np.array([[scalars[self.path]]])

        def get_timestamp(self):
            return datetime.fromtimestamp(time.time() - SHM_CLOCK_OFFSET)

    module = types.ModuleType("dao")
    module.shm = shm
    return module


def fake_config(z2c, hdf5_dir, report_dir):
    """The data grabber config naming fake_dao's shared memories, with only the entries
    the analysis needs."""
    return {
        "acquisition": {"semaphore": 0, "analysed_camera": "camera"},
        "output": {"hdf5_dir": str(hdf5_dir), "report_dir": str(report_dir)},
        "threads": {
            "loop": {"pacer": "wfs_frames", "rate": "wfs_fps", "streams": {
                "wfs_frames": {"shm": "wfs_frames", "dataset": "WFS/WFS_Images",
                               "timestamps": "WFS/WFS_TimeStamps", "keep_every": 100},
                "dm_commands": {"shm": "dm", "dataset": "WFS/DM_commands", "timestamps": "WFS/DM_TimeStamps"},
                "wfs_measurements": {"shm": "meas", "dataset": "WFS/WFS_measurements"},
                "loop_status": {"shm": "loop_cmd", "dataset": "WFS/loop_status"},
            }},
            "camera": {"pacer": "frames", "rate": "sci_fps", "streams": {
                "frames": {"shm": "psf", "dataset": "Science/camera/Science_PSFs",
                           "timestamps": "Science/camera/PSF_TimeStamps"},
            }},
        },
        "static": {
            "wfs_fps": {"source": "wfs_fps.im.shm", "attr": ["WFS/WFS_Images@FPS", "WFS@Loop_Freq"]},
            "dm_map": {"source": "dm_map.im.shm", "dataset": "WFS/DM_Map"},
            "m2c": {"source": "m2c.im.shm", "dataset": "Calibration/M2C"},
            "z2c": {"source": str(z2c), "dataset": "Calibration/Z2C"},
            "loop_gain": {"source": "loop_gain.im.shm", "attr": "WFS@Loop_Gain"},
            "loop_leak": {"source": "loop_leak.im.shm", "attr": "WFS@Loop_Leak"},
            "sci_fps": {"source": "sci_fps.im.shm", "attr": "Science/camera/Science_PSFs@FPS"},
        },
    }


@pytest.fixture
def grabber(tmp_path, monkeypatch, output_config):
    """Set up aott.telemetry against fake_dao; returns set_up(config=None, **fake_dao options),
    which installs `config` (default: fake_config) and returns (telemetry module, config)."""
    hdf5_dir, report_dir = output_config
    source = tmp_path / "source.hdf5"
    write_observation(source)
    z2c = tmp_path / "z2c.npy"
    with h5py.File(source, "r") as f:
        np.save(z2c, f["Calibration/Z2C"][:])

    def set_up(config=None, **options):
        monkeypatch.setitem(sys.modules, "dao", fake_dao(source, **options))
        import aott.telemetry as telemetry

        config = config or fake_config(z2c, hdf5_dir, report_dir)
        monkeypatch.setattr(telemetry, "dao", sys.modules["dao"])
        monkeypatch.setattr(telemetry, "LoadInstrument", lambda: LoadInstrument(CONFIG_DIR / "example_instrument.toml"))
        monkeypatch.setattr(telemetry, "LoadConfig", lambda: config)
        return telemetry, config

    set_up.minimal_config = lambda: fake_config(z2c, hdf5_dir, report_dir)
    set_up.source = source
    return set_up


@requires_typst
def test_observe(grabber, monkeypatch, output_config):
    hdf5_dir, report_dir = output_config
    config = grabber.minimal_config()
    config["threads"]["camera"]["streams"]["frames"]["saturation_level"] = 1e9
    grabber(config)
    import aott.observe as observe

    monkeypatch.setattr(sys, "argv", ["observe", "Test star", "2.5", "--no-simbad"])
    start = datetime.now(timezone.utc)
    observe.main()

    grabbed = list(hdf5_dir.glob("*/Test_star_*.hdf5"))
    assert len(grabbed) == 1
    assert grabbed[0].parent.name == start.strftime("%Y-%m-%d")
    with h5py.File(grabbed[0], "r") as f:
        assert f["WFS/loop_status"][:].all()
        assert f["WFS/Analysis/r0"].shape[0] > 0
    assert len(list(report_dir.glob("*/ao_reportTest_star_*.pdf"))) == 1


def test_required_destinations_suffice(grabber):
    """A grab with only the entries the analysis needs gives a file the three analyses read."""
    telemetry, _ = grabber()
    host_start = time.time()
    path = telemetry.acquire("Test star", 2.5, no_simbad=True)
    analyse(path)
    with h5py.File(path, "r") as f:
        assert f["Science"].attrs["Analysed_Camera"] == "camera"
        assert f["Science/Analysis"].attrs["Camera"] == "camera"
        assert f["Science/Analysis/Long_Exposure/r0"].shape[0] > 0
        assert f["WFS/Analysis/r0"].shape[0] > 0
        # The shm timestamps, not this machine's clock
        assert f["WFS/DM_TimeStamps"][0] < host_start
        assert len(f["WFS/DM_TimeStamps"]) == len(f["WFS/DM_commands"]) == len(f["WFS/loop_status"])


def test_extra_streams_and_cameras(grabber):
    """Extra streams and static entries land at their destinations, with the mean frame over
    every WFS frame read and a saturation flag per camera."""
    config = grabber.minimal_config()
    loop = config["threads"]["loop"]["streams"]
    loop["wfs_frames"]["mean"] = "WFS/Mean_Frame"
    loop["modes"] = {"shm": "dm", "dataset": "WFS/Reconstructed_Modes", "description": "modal coefficients"}
    config["threads"]["camera"]["streams"]["frames"]["saturation_level"] = 1e9
    config["threads"]["camera2"] = {"pacer": "frames", "rate": "sci_fps", "streams": {
        "frames": {"shm": "psf2", "dataset": "Science/camera2/Science_PSFs",
                   "timestamps": "Science/camera2/PSF_TimeStamps", "saturation_level": 100}}}
    config["static"]["modulator"] = {"source": "modulator.im.shm", "dataset": "WFS/Modulator"}
    config["static"]["dit"] = {"value": 0.01, "attr": "Science/camera2/Science_PSFs@Exposure_Time"}
    telemetry, _ = grabber(config)
    path = telemetry.acquire("Test star", 0.5, no_simbad=True)

    with h5py.File(path, "r") as f:
        n = len(f["WFS/DM_TimeStamps"])
        # frame k of the loop holds k + 1; one kept out of 100, the mean over all
        assert f["WFS/WFS_Images"][:, 0, 0].tolist() == list(range(1, n + 1, 100))
        assert f["WFS/WFS_Images"].attrs["Frame_Step"] == 100
        assert np.allclose(f["WFS/Mean_Frame"][:], (n + 1) / 2)
        assert f["WFS/Reconstructed_Modes"].shape == f["WFS/DM_commands"].shape
        assert f["WFS/Reconstructed_Modes"].attrs["Description"] == "modal coefficients"
        assert f["WFS/Modulator"][:].tolist() == [0, 1, 2, 3, 4]
        assert not f["Science/camera/Science_PSFs"].attrs["Saturated"]
        assert f["Science/camera2/Science_PSFs"].attrs["Saturated"]
        assert f["Science/camera2/Science_PSFs"].attrs["Saturation_Level"] == 100
        assert f["Science/camera2/Science_PSFs"].attrs["Exposure_Time"] == 0.01
        assert f["Science/camera2/Science_PSFs"].shape[0] > 0
        # instrument values only for the cameras the instrument config describes
        assert "Wavelength" in f["Science/camera/Science_PSFs"].attrs
        assert "Wavelength" not in f["Science/camera2/Science_PSFs"].attrs

    from aott.AutomaticAnalysis import saturation_summary
    with h5py.File(path, "r") as f:
        summary = saturation_summary(f)
    assert "camera: not saturated" in summary and "camera2: saturated" in summary


def test_acquire_unsliced_background(grabber):
    """Science shms that can't slice and frames that aren't background subtracted: the frames
    and the background are cropped in Python, and the background is subtracted and saved.
    Saturation is checked before the background is subtracted."""
    with h5py.File(grabber.source, "r") as f:
        psf = f["Science/camera/Science_PSFs"][:]
    # Above every background-subtracted frame in the window, below every raw one
    window = psf[:, 2:30, 4:40]
    assert window.max() - window.max(axis=(1, 2)).min() < 40
    level = window.max() + 50
    config = grabber.minimal_config()
    frames = config["threads"]["camera"]["streams"]["frames"]
    frames.update(window={"y": [2, 30], "x": [4, 40]}, sliceable=False, background="background",
                  saturation_level=level)
    config["static"]["background"] = {"source": "psf_background.im.shm", "dataset": "Science/camera/Dark",
                                      "window_of": "camera.frames"}
    telemetry, _ = grabber(config, sliceable=False, background=100)
    path = telemetry.acquire("Test star", 0.3, no_simbad=True)

    with h5py.File(path, "r") as f:
        frames = f["Science/camera/Science_PSFs"][:]
        dark = f["Science/camera/Dark"][:]
        attrs = dict(f["Science/camera/Science_PSFs"].attrs)
    assert frames.dtype == np.float32
    assert frames.shape[1:] == dark.shape == (28, 36)
    assert (dark == 100).all()
    first = frames[0]
    assert any(np.allclose(first, frame[2:30, 4:40]) for frame in psf)
    assert attrs["Window"].tolist() == [2, 30, 4, 40]
    assert attrs["Saturated"] and frames.max() < level


def test_config_problems(grabber):
    config = grabber.minimal_config()
    assert ConfigProblems(config) == []

    def problems(change):
        changed = copy.deepcopy(config)
        change(changed)
        return " ".join(ConfigProblems(changed))

    assert "WFS/DM_Map" in problems(lambda c: c["static"].pop("dm_map"))
    assert "Science/other/Science_PSFs" in problems(lambda c: c["acquisition"].update(analysed_camera="other"))
    assert "keep_every = 1" in problems(
        lambda c: c["threads"]["loop"]["streams"]["dm_commands"].update(keep_every=2))
    assert "same thread" in problems(lambda c: c["threads"]["camera"]["streams"].update(
        loop_status=c["threads"]["loop"]["streams"].pop("loop_status")))
    assert "more than one entry" in problems(
        lambda c: c["static"]["dm_map"].update(dataset="Calibration/M2C"))
    assert "pacer" in problems(lambda c: c["threads"]["loop"].update(pacer="nothing"))
    assert "background" in problems(
        lambda c: c["threads"]["camera"]["streams"]["frames"].update(background="nothing"))
    assert "window_of" in problems(lambda c: c["static"]["dm_map"].update(window_of="loop.nothing"))
    assert "shm is missing" in problems(lambda c: c["threads"]["loop"]["streams"]["dm_commands"].update(shm="TODO"))


@pytest.mark.parametrize("name", ["data_grabber_ekarus.toml", "data_grabber_rama.toml"])
def test_reference_configs_are_complete(name):
    with open(CONFIG_DIR / name, "rb") as f:
        assert ConfigProblems(tomllib.load(f)) == []
