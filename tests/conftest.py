import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import write_observation  # noqa: E402

from aott.Atmosphere_Characterization import Atmosphere_Characterization  # noqa: E402
from aott.PSF_Processing import PSF_Processing  # noqa: E402
from aott.frozen_flow_profiler import profile_file, save_results  # noqa: E402

requires_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not on the PATH")


def analyse(path):
    """The three analyses of AutomaticAnalysis.analyze_and_report, without the report."""
    psf = PSF_Processing(path)
    psf.SetPSFModel()
    psf.AnalyzeAllTheFile()
    Atmosphere_Characterization(path).AnalyzeAllTheFile()
    save_results(path, profile_file(path))


@pytest.fixture(scope="session")
def analysed(tmp_path_factory):
    """A closed-loop synthetic observation, analysed once for the session: (path, truth)."""
    path = tmp_path_factory.mktemp("analysed") / "closed.hdf5"
    truth = write_observation(path)
    analyse(path)
    return path, truth


@pytest.fixture
def output_config(tmp_path, monkeypatch):
    """(hdf5_dir, report_dir) in tmp_path, through a data grabber config that output_dirs reads."""
    hdf5_dir, report_dir = tmp_path / "hdf5", tmp_path / "reports"
    config = tmp_path / "data_grabber.toml"
    config.write_text(f'[output]\nhdf5_dir = "{hdf5_dir.as_posix()}"\nreport_dir = "{report_dir.as_posix()}"\n')
    monkeypatch.setattr("aott.observation_files.DATA_GRABBER_FILE", config)
    return hdf5_dir, report_dir
