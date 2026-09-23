"""Observations that are not simply closed loop throughout."""
import h5py
import numpy as np
import pytest

from conftest import analyse, requires_typst
from synthetic import write_observation

from aott.Atmosphere_Characterization import Atmosphere_Characterization
from aott.AutomaticAnalysis import analyze_and_report
from aott.PSF_Processing import PSF_Processing
from aott.frozen_flow_profiler import ProfilerInputError, profile_file


@requires_typst
def test_open_loop_only(tmp_path):
    path = tmp_path / "open.hdf5"
    write_observation(path, loop_status=np.zeros(3000, dtype=int))
    assert analyze_and_report(path, tmp_path / "reports") is not None
    with h5py.File(path, "r") as f:
        assert f["WFS/Analysis/r0"].shape[0] == 0
        assert "Open_Loop_PSD" in f["WFS/Analysis"]
        assert "Frozen_Flow" not in f["WFS/Analysis"]
        assert f["Science/Analysis/Long_Exposure_OpenLoop/r0"].shape[0] > 0


@requires_typst
def test_open_then_closed(tmp_path):
    path = tmp_path / "mixed.hdf5"
    status = np.r_[np.zeros(1500, dtype=int), np.ones(1500, dtype=int)]
    write_observation(path, loop_status=status)
    assert analyze_and_report(path, tmp_path / "reports") is not None
    with h5py.File(path, "r") as f:
        assert f["WFS/Analysis/r0"].shape[0] > 0
        assert "Open_Loop_PSD" in f["WFS/Analysis"]
        assert f["Science/Analysis/Long_Exposure/r0"].shape[0] > 0
        assert f["Science/Analysis/Long_Exposure_OpenLoop/r0"].shape[0] > 0


def test_loop_status_as_attribute(tmp_path):
    path = tmp_path / "attr.hdf5"
    write_observation(path, duration=1.5, loop_status_in_attr=True)
    analysis = Atmosphere_Characterization(path)
    assert analysis.is_closed_loop_per_sample.all()


def test_no_loop_status(tmp_path):
    path = tmp_path / "no_status.hdf5"
    write_observation(path, duration=1.5, loop_status=False)
    with pytest.raises(KeyError):
        Atmosphere_Characterization(path)
    with pytest.raises(ProfilerInputError):
        profile_file(path)


def test_unknown_elevation(tmp_path):
    path = tmp_path / "no_simbad.hdf5"
    write_observation(path, elevation=np.nan)
    analyse(path)
    with h5py.File(path, "r") as f:
        assert np.isnan(f["WFS/Analysis/r0_Zenith"][:]).all()
        assert np.isfinite(f["WFS/Analysis/r0"][:]).all()


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="PSF_Processing and AnalysisViewer need the Science group (repo review W20, action 8)")
def test_no_science_camera(tmp_path):
    path = tmp_path / "no_science.hdf5"
    write_observation(path, science=False)
    analyze_and_report(path, tmp_path / "reports")


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="One NaN science frame stops the PSF analysis (repo review W18, action 7)")
def test_nan_science_frame(tmp_path):
    path = tmp_path / "nan_frame.hdf5"
    write_observation(path, duration=1.5)
    with h5py.File(path, "a") as f:
        f["Science/Science_PSFs"][5] = np.nan
    psf = PSF_Processing(path)
    psf.SetPSFModel()
    psf.AnalyzeAllTheFile()


def test_science_camera_rate_from_its_timestamps(tmp_path):
    """The science camera is not synchronized with the loop: its batches follow
    PSF_TimeStamps, even when the FPS attr disagrees."""
    path = tmp_path / "wrong_fps.hdf5"
    truth = write_observation(path, duration=1.5)
    with h5py.File(path, "a") as f:
        f["Science/Science_PSFs"].attrs["FPS"] = 1000
    psf = PSF_Processing(path)
    assert psf.fps == pytest.approx(truth["science_fps"])
    assert psf.batch_size == round(psf.batch_duration * truth["science_fps"])
