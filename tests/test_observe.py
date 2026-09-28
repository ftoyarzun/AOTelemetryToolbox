"""python -m aott.observe end to end, with a fake dao that replays a synthetic observation."""
import sys
import time
import types
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

from conftest import requires_typst
from synthetic import write_observation

from aott.config import CONFIG_DIR, LoadInstrument


def fake_dao(source, sliceable=True, background=0):
    """A dao module whose shared memories replay `source`: the loop at its Loop_Freq, the
    science camera at its FPS, with the loop closed. The science shms add `background` to
    the frames and, unless `sliceable`, refuse a window."""
    with h5py.File(source, "r") as f:
        data = dict(wfs_frames=f["WFS/WFS_Images"][:], dm=f["WFS/DM_commands"][:], meas=f["WFS/WFS_measurements"][:],
                    psf=f["Science/Science_PSFs"][:], pup=f["WFS/Valid_Pixel_Map"][:], dm_map=f["WFS/DM_Map"][:],
                    m2c=f["Calibration/M2C"][:])
        loop_fps = float(f["WFS"].attrs["Loop_Freq"])
        science_fps = float(f["Science/Science_PSFs"].attrs["FPS"])
    scalars = dict(loop_gain=0.5, loop_leak=0.99, wfs_fps=loop_fps, wfs_gain=1, sci_dit=1 / science_fps,
                   sci_fps=science_fps, sci_gain=1, loop_cmd=1)
    count = {"wfs": 0, "psf": 0}
    due = {"wfs": time.perf_counter(), "psf": time.perf_counter()}

    def wait(key, period):
        # Busy wait: time.sleep is too coarse for 1 kHz on Windows
        due[key] += period
        while time.perf_counter() < due[key]:
            pass
        count[key] += 1

    class shm:
        def __init__(self, path):
            self.path = path

        def get_data(self, check=False, semNb=None, **window):
            if self.path == "wfs_frames":
                if check:
                    wait("wfs", 1 / loop_fps)
                return data["wfs_frames"][count["wfs"] % len(data["wfs_frames"])]
            if self.path in ("dm", "meas"):
                return data[self.path][count["wfs"] % len(data[self.path])]
            if self.path in ("psf", "psf_background.im.shm"):
                if window and not sliceable:
                    raise TypeError("this shm can't slice")
                if self.path == "psf_background.im.shm":
                    frame = np.full(data["psf"].shape[1:], background)
                else:
                    if check:
                        wait("psf", 1 / science_fps)
                    frame = data["psf"][count["psf"] % len(data["psf"])] + background
                return frame[window["y"], window["x"]] if window else frame
            if self.path in ("pup", "dm_map", "m2c"):
                return data[self.path]
            return np.array([[scalars[self.path]]])

    module = types.ModuleType("dao")
    module.shm = shm
    return module


def fake_config(z2c, hdf5_dir, report_dir):
    """The data grabber config naming fake_dao's shared memories."""
    return {
        "shm": {"wfs": {"frames": "wfs_frames", "valid_pixel_map": "pup", "measurements": "meas",
                        "fps": "wfs_fps", "gain": "wfs_gain"},
                "dm": {"commands": "dm", "m2c": "m2c", "dm_map": "dm_map"},
                "loop": {"cmd": "loop_cmd", "gain": "loop_gain", "leak": "loop_leak"},
                "science": {"frames": "psf", "dit": "sci_dit", "fps": "sci_fps", "gain": "sci_gain"}},
        "acquisition": {"semaphore": 0, "wfs_frame_step": 100},
        "calibration": {"Z2C": str(z2c)},
        "output": {"hdf5_dir": str(hdf5_dir), "report_dir": str(report_dir)},
    }


@requires_typst
def test_observe(tmp_path, monkeypatch, output_config):
    hdf5_dir, report_dir = output_config
    source = tmp_path / "source.hdf5"
    write_observation(source)
    z2c = tmp_path / "z2c.npy"
    with h5py.File(source, "r") as f:
        np.save(z2c, f["Calibration/Z2C"][:])

    monkeypatch.setitem(sys.modules, "dao", fake_dao(source))
    import aott.telemetry as telemetry
    import aott.observe as observe

    monkeypatch.setattr(telemetry, "dao", sys.modules["dao"])
    monkeypatch.setattr(telemetry, "LoadInstrument", lambda: LoadInstrument(CONFIG_DIR / "example_instrument.toml"))
    monkeypatch.setattr(telemetry, "LoadConfig", lambda: fake_config(z2c, hdf5_dir, report_dir))
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


def test_acquire_unsliced_background(tmp_path, monkeypatch, output_config):
    """Science shms that can't slice and frames that aren't background subtracted: the frames
    and the background are cropped in Python, and the background is subtracted and saved."""
    hdf5_dir, report_dir = output_config
    source = tmp_path / "source.hdf5"
    write_observation(source)
    z2c = tmp_path / "z2c.npy"
    with h5py.File(source, "r") as f:
        np.save(z2c, f["Calibration/Z2C"][:])
        psf = f["Science/Science_PSFs"][:]

    monkeypatch.setitem(sys.modules, "dao", fake_dao(source, sliceable=False, background=100))
    import aott.telemetry as telemetry

    config = fake_config(z2c, hdf5_dir, report_dir)
    config["shm"]["science"]["background"] = "psf_background.im.shm"
    config["acquisition"].update(science_background_subtracted=False, science_shm_sliceable=False,
                                 science_crop={"y": [2, 30], "x": [4, 40]})
    monkeypatch.setattr(telemetry, "dao", sys.modules["dao"])
    monkeypatch.setattr(telemetry, "LoadInstrument", lambda: LoadInstrument(CONFIG_DIR / "example_instrument.toml"))
    monkeypatch.setattr(telemetry, "LoadConfig", lambda: config)
    path = telemetry.acquire("Test star", 0.3, no_simbad=True)

    with h5py.File(path, "r") as f:
        frames = f["Science/Science_PSFs"][:]
        dark = f["Science/Dark"][:]
    assert frames.dtype == np.float32
    assert frames.shape[1:] == dark.shape == (28, 36)
    assert (dark == 100).all()
    first = frames[0]
    assert any(np.allclose(first, frame[2:30, 4:40]) for frame in psf)


def test_acquire_controlled_modes(tmp_path, monkeypatch, output_config):
    """The number of controlled modes defaults to the columns of M2C, and can be given."""
    hdf5_dir, report_dir = output_config
    source = tmp_path / "source.hdf5"
    write_observation(source)
    z2c = tmp_path / "z2c.npy"
    with h5py.File(source, "r") as f:
        np.save(z2c, f["Calibration/Z2C"][:])
        n_modes = f["Calibration/M2C"].shape[1]

    monkeypatch.setitem(sys.modules, "dao", fake_dao(source))
    import aott.telemetry as telemetry

    monkeypatch.setattr(telemetry, "dao", sys.modules["dao"])
    monkeypatch.setattr(telemetry, "LoadInstrument", lambda: LoadInstrument(CONFIG_DIR / "example_instrument.toml"))
    monkeypatch.setattr(telemetry, "LoadConfig", lambda: fake_config(z2c, hdf5_dir, report_dir))
    for given, saved in ((None, n_modes), (100, 100)):
        path = telemetry.acquire("Test star", 0.2, no_simbad=True, controlled_modes=given)
        with h5py.File(path, "r") as f:
            assert f["Calibration"].attrs["Total_Number_Of_Controlled_Modes"] == saved
        path.unlink()
    with pytest.raises(SystemExit, match="controlled modes"):
        telemetry.acquire("Test star", 0.2, no_simbad=True, controlled_modes=n_modes + 1)
