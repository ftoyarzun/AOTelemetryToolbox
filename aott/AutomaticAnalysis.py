from aott.PSF_Processing import PSF_Processing
from aott.Atmosphere_Characterization import Atmosphere_Characterization
from aott.AnalysisViewer import AnalysisViewer
from aott.AllSkyCamGrabber import GrabAllSkyFrame
from aott.Telemetry_conversion import RunConversion
from pathlib import Path
import subprocess
import numpy as np
import pylab as plt
import shutil

from datetime import datetime


DATE = "/" + datetime.now().strftime("%Y-%m-%d")
ROOT_FOLDER = r"/mnt/papydisk"
ROOT_FOLDER_RSYNC = r"/home/daouser/DAODATA/ao_reports/data"
ANALYSIS = "/Analysis"

# GrabAllSkyFrame()

# DATE = '/2026-01-21'
# RunConversion(DATE)


analysis_folder = Path(ROOT_FOLDER + ANALYSIS + DATE)
latest_file = max(analysis_folder.iterdir(), key=lambda f: f.stat().st_mtime)
print(latest_file)


p2 = PSF_Processing(latest_file, batch_size=100)
p2.ProcessDark()
p2.SetSkyOrCalibContiditons()
p2.SetPSFModel()
p2.AnalyzeAllTheFile()


atm_char = Atmosphere_Characterization(latest_file, batch_size=4000, filter_TT=False)
atm_char.AnalyzeAllTheFile()


av = AnalysisViewer(latest_file)

av.CreateAtmosphericAnalysisFigures()
av.CreatePSFAnalysisFigures()


import subprocess

file_title = str(latest_file).split("/")[-1].split(".hdf5")[0]
target_name = file_title.split("-")[0]

cmd = [
    "typst",
    "compile",
    "ao_report.typ",
    "ao_report" + file_title + ".pdf",
    "--input",
    "AOtitle=" + str(latest_file).split("/")[-1].split(".hdf5")[0],
    "--input",
    "telescope=T152-Papyrus",
    "--input",
    "date=" + DATE,
    "--input",
    "target=" + target_name,
    "--input",
    f"elevation={p2.elevation:.1f}",
    "--input",
    "loop_gain=" + str(atm_char.loop_gain),
    "--input",
    "loop_leak=" + str(atm_char.loop_leak),
    "--input",
    "loop_freq=" + str(atm_char.freq),
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
except subprocess.CalledProcessError as e:
    print("STDOUT:")
    print(e.stdout)
    print("\nSTDERR:")
    print(e.stderr)

print("ao_report" + file_title + ".pdf")


report_file_name = Path("ao_report" + file_title + ".pdf")
save_folder = Path("/mnt/papydisk/AutomaticReports"+ DATE)
save_folder.mkdir(parents=True, exist_ok=True)

save_folder_rsync = Path(ROOT_FOLDER_RSYNC + DATE)
save_folder_rsync.mkdir(parents=True, exist_ok=True)

shutil.copy(str(report_file_name), str(save_folder_rsync / report_file_name.name))
shutil.move(str(report_file_name), str(save_folder / report_file_name.name))



