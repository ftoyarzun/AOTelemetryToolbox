"""The analyses and the reports on a closed-loop synthetic observation of one known layer."""
import shutil
from datetime import datetime

import h5py
import numpy as np
import pytest

from conftest import requires_typst

from aott.AnalysisViewer import AnalysisViewer
from aott.AutomaticAnalysis import analyze_and_report
from aott.NightlyReport import NightlyReport
from aott.config import AnalysisSettings
from aott.report import compile_report, new_run_dir


def test_telemetry_r0(analysed):
    path, truth = analysed
    with h5py.File(path, "r") as f:
        r0_cm = f["WFS/Analysis/r0"][:]
    assert np.median(r0_cm) / 100 == pytest.approx(truth["r0"], rel=0.15)


def test_autocorrelation_v0_and_tau0(analysed):
    path, truth = analysed
    with h5py.File(path, "r") as f:
        v0 = np.median(f["WFS/Analysis/V0_Autocorrelation"][:])
        tau0_ms = np.median(f["WFS/Analysis/tau0_Autocorrelation"][:])
    assert v0 == pytest.approx(truth["speed"], rel=0.15)
    # 0.31 r0 / V0, against the truth's 0.314 r0 / V
    assert tau0_ms / 1000 == pytest.approx(truth["tau0"], rel=0.25)


def test_frozen_flow_finds_the_layer(analysed):
    path, truth = analysed
    with h5py.File(path, "r") as f:
        speed = f["WFS/Analysis/Frozen_Flow/Speed"][0]
        direction = f["WFS/Analysis/Frozen_Flow/Direction"][0]
    found = (np.abs(speed - truth["speed"]) < 0.2 * truth["speed"]) & (np.abs(direction - truth["direction_deg"]) < 15)
    assert found.any()


@pytest.mark.xfail(strict=True, reason="The profiler also fits a strong static (0 m/s) layer on single-layer "
                                       "data, which pulls V0 low (repo review W13, action 3)")
def test_frozen_flow_v0(analysed):
    path, truth = analysed
    with h5py.File(path, "r") as f:
        v0 = f["WFS/Analysis/Frozen_Flow/V0"][0]
    assert v0 == pytest.approx(truth["speed"], rel=0.25)


def test_analysis_groups(analysed):
    path, truth = analysed
    zenith_factor = np.sin(np.radians(truth["elevation"])) ** (-3 / 5)
    atmosphere = AnalysisSettings("atmosphere")
    with h5py.File(path, "r") as f:
        wfs = f["WFS/Analysis"]
        assert "tau0" not in wfs and "V0" not in wfs
        assert wfs["r0_Zenith"][:] == pytest.approx(wfs["r0"][:] * zenith_factor)
        assert wfs["tau0_Autocorrelation_Zenith"][:] == pytest.approx(wfs["tau0_Autocorrelation"][:] * zenith_factor)
        assert wfs["Seeing_Zenith"][:] == pytest.approx(wfs["Seeing"][:] / zenith_factor)
        for name in ("L0", "Effective_Gain", "Measured_Loop_Delay", "Loop_Bandwidth"):
            assert not wfs[name].attrs["Validated"]
        assert wfs.attrs["Batch_Duration_s"] == atmosphere["batch_duration"]
        assert wfs.attrs["N_Zernike"] == atmosphere["n_zernike"]
        ff = wfs["Frozen_Flow"]
        assert ff["r0_Zenith"][:] == pytest.approx(ff["r0"][:] * zenith_factor)
        assert ff["tau0_Zenith"][:] == pytest.approx(ff["tau0"][:] * zenith_factor)
        assert ff["Seeing_Zenith"][:] == pytest.approx(ff["Seeing"][:] / zenith_factor)

        long_exposure = f["Science/Analysis/Long_Exposure"]
        assert long_exposure["r0_Zenith"][:] == pytest.approx(long_exposure["r0"][:] * zenith_factor)
        assert long_exposure["sr_otf"].attrs["Obstruction_ratio"] == f["Calibration"].attrs["Obstruction_ratio"]
        assert long_exposure["sr_fit"].attrs["Units"] == "%"
        assert long_exposure["Seeing_Zenith"][:] == pytest.approx(long_exposure["Seeing"][:] / zenith_factor)
        assert f["Science/Analysis"].attrs["Batch_Duration_s"] == AnalysisSettings("psf")["batch_duration"]
        assert f["Science/Analysis"].attrs["Frame_Rate_Hz"] == pytest.approx(truth["science_fps"])


def test_center_of_gravity_is_continuous_over_the_run(analysed):
    path, truth = analysed
    with h5py.File(path, "r") as f:
        cog_x = f["Science/Analysis/Short_Exposure/CoG-X"][:]
    batch = int(round(AnalysisSettings("psf")["batch_duration"] * truth["science_fps"]))
    # One mean removed per run (the whole file here), not one per batch
    assert np.mean(cog_x) == pytest.approx(0, abs=1e-9)
    batch_means = [np.mean(cog_x[k:k + batch]) for k in range(0, len(cog_x), batch)]
    assert np.max(np.abs(batch_means)) > 1e-6


def test_viewer_figures(analysed, tmp_path):
    path, _ = analysed
    viewer = AnalysisViewer(path, figure_dir=tmp_path)
    viewer.CreateAtmosphericAnalysisFigures()
    viewer.CreatePSFAnalysisFigures()
    figures = viewer.manifest["figures"]
    for key in ("r0", "tau0", "V0", "psd_comparison", "frozen_flow", "sr", "psf_frames", "jitter"):
        assert figures.get(key), key
    for key in ("L0", "loop_params", "loop_bandwidth"):
        assert key not in figures
    stats = viewer.manifest["stats"]
    with h5py.File(path, "r") as f:
        assert stats["r0_wfs_n"] == f["WFS/Analysis/r0"].shape[0]
    assert viewer.manifest["settings"]["psf_batch_s"] == AnalysisSettings("psf")["batch_duration"]
    assert all(np.isfinite(v) for v in stats.values())


def test_compile_report_without_typst(analysed, monkeypatch):
    path, _ = analysed
    run_dir = new_run_dir("test_report_")
    viewer = AnalysisViewer(path, figure_dir=run_dir)
    viewer.SaveFigureManifest()
    monkeypatch.setattr("aott.report.shutil.which", lambda name: None)
    assert compile_report("ao_report.typ", run_dir, run_dir / "never.pdf") is None
    assert run_dir.is_dir()
    shutil.rmtree(run_dir)


@requires_typst
def test_analyze_and_report(analysed, tmp_path):
    path, _ = analysed
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    pdf = analyze_and_report(copy, tmp_path / "reports")
    assert pdf is not None and pdf.exists()
    assert pdf.parent.parent == tmp_path / "reports"


@requires_typst
def test_nightly_report(analysed, tmp_path):
    path, _ = analysed
    with h5py.File(path, "r") as f:
        end = datetime.fromtimestamp(float(f["WFS/DM_TimeStamps"][-1]) + 3600)
    (tmp_path / "figures").mkdir()
    report = NightlyReport(path.parent, window_hours=5, end_time=end, figure_dir=tmp_path / "figures")
    assert [o["file"] for o in report.observations] == [path.name]
    assert report.telescope == "Synthetic-Synthetic"
    assert "r0_wfs" in report.observations[0]["stats"]
    report.CreateFigures()
    pdf = report.CompileReport(tmp_path / "reports")
    assert pdf is not None and pdf.exists()
