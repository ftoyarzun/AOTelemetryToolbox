from aott.PSF_Processing import PSF_Processing
from aott.Atmosphere_Characterization import Atmosphere_Characterization
from aott.AnalysisViewer import AnalysisViewer
from aott.config import DATA_GRABBER_FILE
from pathlib import Path
import subprocess
import numpy as np
import pylab as plt
import shutil

from datetime import datetime

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


DATE = "/" + datetime.now().strftime("%Y-%m-%d")

# Where to find the newest HDF5 file and where to write the compiled PDF --
# read from the [output] section of config/data_grabber.toml, so this
# script and aott/telemetry.py agree on these paths.
with open(DATA_GRABBER_FILE, "rb") as _f:
    _output_config = tomllib.load(_f)["output"]
hdf5_dir = Path(_output_config["hdf5_dir"])
report_dir = Path(_output_config["report_dir"])

# GrabAllSkyFrame()

# DATE = '/2026-01-21'
# RunConversion(DATE)


# hdf5_dir holds one dated subfolder per day of observations; fall back to
# hdf5_dir itself if that subfolder doesn't exist, e.g. local test data
# sitting directly in hdf5_dir with no date structure.
analysis_folder = hdf5_dir / DATE.lstrip("/")
if not analysis_folder.is_dir():
    analysis_folder = hdf5_dir
latest_file = max(analysis_folder.iterdir(), key=lambda f: f.stat().st_mtime)
print(latest_file)


# Frames skipped after each open/closed-loop transition while the loop settles
# (science frames for PSF_Processing, loop iterations for Atmosphere_Characterization)
PSF_TRANSITION_BUFFER = 10
WFS_TRANSITION_BUFFER = 10

p2 = PSF_Processing(latest_file, batch_duration=0.1, transition_buffer=PSF_TRANSITION_BUFFER)
p2.SetPSFModel()
p2.AnalyzeAllTheFile()


atm_char = Atmosphere_Characterization(latest_file, batch_duration=4.0, filter_TT=False,
                                       transition_buffer=WFS_TRANSITION_BUFFER)
atm_char.AnalyzeAllTheFile()


av = AnalysisViewer(latest_file)

av.CreateAtmosphericAnalysisFigures()
av.CreatePSFAnalysisFigures()
av.SaveFigureManifest()


file_title = latest_file.stem
# Read from the HDF5 file itself (Science.attrs["Target"]) rather than
# guessed from the filename -- p2 has already read it in during __init__.
target_name = str(p2.target_name)

# Dropped in later by hand at the repo root; "none" tells the template to
# leave the logo slot out of the header instead of trying to load it.
logo_path = Path("logo.png")

cmd = [
    "typst",
    "compile",
    "ao_report.typ",
    "ao_report" + file_title + ".pdf",
    "--input",
    "AOtitle=" + file_title,
    "--input",
    "telescope=T152-Papyrus",
    "--input",
    "date=" + DATE,
    "--input",
    "target=" + target_name,
    "--input",
    f"elevation={p2.elevation:.1f}",
    "--input",
    f"loop_gain={atm_char.loop_gain:.3f}",
    "--input",
    f"loop_leak={atm_char.loop_leak:.3f}",
    "--input",
    f"loop_freq={atm_char.freq:.1f}",
    "--input",
    "logo=" + (logo_path.name if logo_path.exists() else "none"),
]

if av.VMag:
    cmd += [
        "--input",
        f"VMag={av.VMag:.2f}",
        "--input",
        f"RMag={av.RMag:.2f}",
        "--input",
        f"JMag={av.JMag:.2f}",
        "--input",
        f"HMag={av.HMag:.2f}",
    ]
else:
    cmd += [
        "--input",
        "VMag=none",
        "--input",
        "RMag=none",
        "--input",
        "JMag=none",
        "--input",
        "HMag=none",
    ]

try:
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    # Typst has embedded each PNG directly in the PDF by this point, so the
    # standalone files are no longer needed. Left in place on a failed
    # compile, since they're useful for debugging what went wrong.
    av.RemoveFigureFiles()
except subprocess.CalledProcessError as e:
    print("STDOUT:")
    print(e.stdout)
    print("\nSTDERR:")
    print(e.stderr)

print("ao_report" + file_title + ".pdf")


report_file_name = Path("ao_report" + file_title + ".pdf")
save_folder = report_dir / DATE.lstrip("/")
save_folder.mkdir(parents=True, exist_ok=True)

shutil.move(str(report_file_name), str(save_folder / report_file_name.name))



