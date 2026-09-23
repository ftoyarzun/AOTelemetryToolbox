"""
Where the project's config files are, and how to read the instrument and analysis ones.
There is exactly one of each, all in the config/ folder at the repo root:

    config/data_grabber.toml   this machine: shared memories, calibration files, output folders
    config/instrument.toml     the instrument: site, pupil, DM, WFS, science camera
    config/analysis.toml       the analysis: batch lengths, transition buffers, fit settings

Their paths come from the package location, so they don't depend on the folder the scripts
are run from. The other files in config/ are filled-in reference copies that no code reads.
"""
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


def LoadInstrument(path=INSTRUMENT_FILE):
    """Read the instrument TOML file and exit if a value is still "TODO"."""

    with open(path, "rb") as f:
        instrument = tomllib.load(f)

    missing = [f"{section}.{key}"
               for section, values in instrument.items()
               for key, value in values.items() if value == "TODO"]
    if missing:
        sys.exit(f"{path}: fill in these values first: " + ", ".join(missing))

    return instrument
