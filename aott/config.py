"""
Where the project's config files are, and how to read the instrument and analysis ones.
There is exactly one of each, all in the config/ folder at the repo root:

    config/data_grabber.toml   this machine: the threads and shared memories to record, output folders
    config/instrument.toml     the instrument: site, pupil, DM, WFS, science cameras
    config/analysis.toml       the analysis: batch lengths, transition buffers, fit settings

Their paths come from the package location, so they don't depend on the folder the scripts
are run from. The other files in config/ are filled-in reference copies that no code reads.
"""
import os
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DATA_GRABBER_FILE = CONFIG_DIR / "data_grabber.toml"
INSTRUMENT_FILE = CONFIG_DIR / "instrument.toml"
ANALYSIS_FILE = CONFIG_DIR / "analysis.toml"


def AnalysisSettings(section, **given):
    """The [section] settings of config/analysis.toml, overridden by every keyword in
    `given` that isn't None."""

    with open(ANALYSIS_FILE, "rb") as f:
        settings = tomllib.load(f)[section]
    settings.update({key: value for key, value in given.items() if value is not None})
    return settings


def PinToCPUs(path=DATA_GRABBER_FILE):
    """Restrict this process to the [acquisition] cpus of the data grabber config, if given.

    Only threads started afterwards inherit the affinity, and the BLAS libraries start
    their thread pools (and read the *_NUM_THREADS variables) when numpy is imported,
    so call this before importing numpy or any aott module other than this one.
    Child processes (the typst compile) inherit it too. No-op without the setting,
    or where sched_setaffinity doesn't exist (Windows).
    """

    with open(path, "rb") as f:
        cpus = tomllib.load(f).get("acquisition", {}).get("cpus")
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return

    os.sched_setaffinity(0, cpus)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, str(len(cpus)))


def LoadInstrument(path=INSTRUMENT_FILE):
    """Read the instrument TOML file and exit if a value is still "TODO"."""

    with open(path, "rb") as f:
        instrument = tomllib.load(f)

    def Todo(table, prefix):
        for key, value in table.items():
            if isinstance(value, dict):
                yield from Todo(value, f"{prefix}{key}.")
            elif value == "TODO":
                yield f"{prefix}{key}"

    missing = list(Todo(instrument, ""))
    if missing:
        sys.exit(f"{path}: fill in these values first: " + ", ".join(missing))

    return instrument
