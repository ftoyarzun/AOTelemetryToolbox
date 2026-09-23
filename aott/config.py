"""
Where the project's config files are, and how to read the instrument one. There is exactly
one of each, both in the config/ folder at the repo root:

    config/data_grabber.toml   this machine: shared memories, calibration files, output folders
    config/instrument.toml     the instrument: site, pupil, DM, WFS, science camera

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
