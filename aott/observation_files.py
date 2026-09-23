"""
Finding observation HDF5 files: the [output] folders of config/data_grabber.toml,
the UTC date folders telemetry.py writes into, and the newest observation in them.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

from aott.config import DATA_GRABBER_FILE

HDF5_SUFFIXES = (".hdf5", ".h5")


def output_dirs():
    """(hdf5_dir, report_dir) from the [output] section of config/data_grabber.toml;
    exits while either is still "TODO"."""
    with open(DATA_GRABBER_FILE, "rb") as f:
        output = tomllib.load(f).get("output", {})
    missing = [key for key in ("hdf5_dir", "report_dir") if output.get(key, "TODO") == "TODO"]
    if missing:
        sys.exit(f"{DATA_GRABBER_FILE}: fill in these values first: "
                 + ", ".join(f"output.{key}" for key in missing))
    return Path(output["hdf5_dir"]), Path(output["report_dir"])


def utc_date(timestamp):
    """YYYY-MM-DD of a Unix timestamp, in UTC: the name of its folder in hdf5_dir."""
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")


def observation_span(file):
    """
    (start, end) Unix timestamps of an open observation file: the first and
    last WFS/DM_TimeStamps sample, falling back to Science's PSF_TimeStamps
    dataset, or attr in older files. Only two samples are read, so this stays
    cheap however long the observation is. None if the file has no timestamps.
    """
    for path in ("WFS/DM_TimeStamps", "Science/PSF_TimeStamps"):
        if path in file:
            ts = file[path]
            if ts.shape[0] > 0:
                return float(ts[0]), float(ts[-1])
    if "Science" in file and "PSF_TimeStamps" in file["Science"].attrs:
        ts = np.asarray(file["Science"].attrs["PSF_TimeStamps"], dtype=float)
        if ts.size > 0:
            return float(ts[0]), float(ts[-1])
    return None


def telescope_name(file):
    """'Telescope-Instrument' from the root attrs of an open observation file
    (either alone if the other is missing), or None if it has neither."""
    names = [str(file.attrs[key]) for key in ("Telescope", "Instrument") if key in file.attrs]
    return "-".join(names) or None


def date_folders(hdf5_dir, start, end):
    """
    The existing hdf5_dir/<UTC date> folders for every UTC date from Unix time
    `start` to `end`, or [hdf5_dir] itself if none exists (a flat folder of
    test data).
    """
    hdf5_dir = Path(hdf5_dir)
    folders = []
    day = datetime.fromtimestamp(start, timezone.utc).date()
    while day <= datetime.fromtimestamp(end, timezone.utc).date():
        folder = hdf5_dir / day.strftime("%Y-%m-%d")
        if folder.is_dir():
            folders.append(folder)
        day += timedelta(days=1)
    return folders or [hdf5_dir]


def hdf5_files(folders):
    """Every .hdf5/.h5 file directly in `folders`, sorted by name."""
    return sorted(f for folder in folders for f in Path(folder).iterdir()
                  if f.is_file() and f.suffix.lower() in HDF5_SUFFIXES)


def newest_file(hdf5_dir, unanalysed_only=False):
    """
    The observation with the latest start time (observation_span) in today's
    and yesterday's UTC date folders of hdf5_dir. With `unanalysed_only`, only
    files without a WFS/Analysis group count. Files that can't be opened or
    have no timestamps are skipped. Exits if no file qualifies.
    """
    now = datetime.now(timezone.utc).timestamp()
    folders = date_folders(hdf5_dir, now - 86400, now)
    newest, newest_start = None, None
    for path in hdf5_files(folders):
        try:
            with h5py.File(path, "r") as file:
                if unanalysed_only and "WFS/Analysis" in file:
                    continue
                span = observation_span(file)
        except OSError:
            continue
        if span is not None and (newest_start is None or span[0] > newest_start):
            newest, newest_start = path, span[0]
    if newest is None:
        what = "not yet analysed " if unanalysed_only else ""
        sys.exit(f"No {what}observation file in " + ", ".join(str(f) for f in folders))
    return newest
