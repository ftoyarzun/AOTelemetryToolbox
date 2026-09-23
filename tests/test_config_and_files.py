"""Analysis settings, the [output] folders and the choice of the newest observation."""
import time
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

from aott.config import AnalysisSettings
from aott.observation_files import newest_file, output_dirs


def test_analysis_settings_sections_and_overrides():
    for section in ("psf", "atmosphere", "frozen_flow"):
        assert AnalysisSettings(section)
    configured = AnalysisSettings("atmosphere")["batch_duration"]
    assert AnalysisSettings("atmosphere", batch_duration=None)["batch_duration"] == configured
    assert AnalysisSettings("atmosphere", batch_duration=configured + 1)["batch_duration"] == configured + 1


def test_output_dirs_exits_while_todo(tmp_path, monkeypatch):
    config = tmp_path / "data_grabber.toml"
    config.write_text('[output]\nhdf5_dir = "TODO"\nreport_dir = "reports"\n')
    monkeypatch.setattr("aott.observation_files.DATA_GRABBER_FILE", config)
    with pytest.raises(SystemExit, match="output.hdf5_dir"):
        output_dirs()


def _observation(path, start, analysed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["WFS/DM_TimeStamps"] = start + np.arange(10)
        if analysed:
            f["WFS"].create_group("Analysis")


def test_newest_file_by_start_time_in_utc_date_folders(tmp_path):
    now = time.time()

    def folder(t):
        return tmp_path / datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")

    _observation(folder(now - 80000) / "yesterday.hdf5", now - 80000)
    _observation(folder(now) / "unanalysed.hdf5", now - 600)
    _observation(folder(now) / "analysed.hdf5", now - 60, analysed=True)
    _observation(tmp_path / "2020-01-01" / "old.hdf5", now - 9e7)
    (folder(now) / "broken.hdf5").write_text("not HDF5")
    (folder(now) / "notes.txt").write_text("not an observation")

    assert newest_file(tmp_path).name == "analysed.hdf5"
    assert newest_file(tmp_path, unanalysed_only=True).name == "unanalysed.hdf5"


def test_newest_file_in_a_flat_folder(tmp_path):
    _observation(tmp_path / "a.hdf5", time.time() - 100)
    _observation(tmp_path / "b.hdf5", time.time() - 10)
    assert newest_file(tmp_path).name == "b.hdf5"


def test_newest_file_exits_when_nothing_qualifies(tmp_path):
    _observation(tmp_path / "a.hdf5", time.time(), analysed=True)
    with pytest.raises(SystemExit):
        newest_file(tmp_path, unanalysed_only=True)
