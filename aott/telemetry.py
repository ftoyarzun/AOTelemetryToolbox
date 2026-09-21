import h5py
import dao
import numpy as np
import pylab as plt
import time
import datetime
from pathlib import Path
from astropy.io import fits
import sys
from aott.PSF_Processing import PSF_Processing
from aott.Atmosphere_Characterization import Atmosphere_Characterization
from aott.AnalysisViewer import AnalysisViewer
from aott.DataGrabber import LoadConfig, Stream, RecordInParallel
import shutil
import argparse

parser = argparse.ArgumentParser(description="Grab telemetry and PSFs, analyze them and build the report.")
parser.add_argument("target", type=str, help="name of the target")
parser.add_argument("duration", type=float, help="acquisition time in seconds")
parser.add_argument("config", help="data grabber TOML file, see data_grabbers/example_data_grabber.toml")
args = parser.parse_args()

config = LoadConfig(args.config)
shm_paths = config["shm"]

WFS_FRAMES_SHM = dao.shm(shm_paths["wfs"]["frames"])
WFS_PUP = dao.shm(shm_paths["wfs"]["valid_pixel_map"]).get_data()
DM_SHM = dao.shm(shm_paths["dm"]["commands"])
M2C_SHM = dao.shm(shm_paths["dm"]["m2c"])
MODAL_VECTOR_INFERENCE_SHM = dao.shm(shm_paths["wfs"]["measurements"])
PSF_IM_CAL = dao.shm(shm_paths["science"]["frames"])
LOOP_CMD_SHM = dao.shm(shm_paths["loop"]["cmd"])
LOOP_GAIN_SHM = dao.shm(shm_paths["loop"]["gain"])
LOOP_LEAK_SHM = dao.shm(shm_paths["loop"]["leak"])

WFS_FPS_SHM = dao.shm(shm_paths["wfs"]["fps"])
WFS_GAIN_SHM = dao.shm(shm_paths["wfs"]["gain"])

SCI_DIT_SHM = dao.shm(shm_paths["science"]["dit"])
SCI_FPS_SHM = dao.shm(shm_paths["science"]["fps"])
SCI_GAIN_SHM = dao.shm(shm_paths["science"]["gain"])

SEM_NB = config["acquisition"]["semaphore"]
WFS_FRAME_STEP = config["acquisition"].get("wfs_frame_step", 1)

# Optional window of the science frame to read, empty means the whole frame
crop = config["acquisition"].get("science_crop")
PSF_WINDOW = {"y": slice(*crop["y"]), "x": slice(*crop["x"])} if crop else {}


start_time = time.monotonic()

wfs_rec, psf_rec = RecordInParallel(
    [
        [Stream(DM_SHM), Stream(WFS_FRAMES_SHM, keep_every=WFS_FRAME_STEP), Stream(MODAL_VECTOR_INFERENCE_SHM)],
        [Stream(PSF_IM_CAL, window=PSF_WINDOW), Stream(LOOP_CMD_SHM)],
    ],
    args.duration,
    SEM_NB,
)

elapsed = time.monotonic() - start_time

print(f"Acquisition finished in {elapsed:.3f} s")
print(f"WFS Camera: {len(wfs_rec.timestamps)} frames")
print(f"PSF Camera: {len(psf_rec.timestamps)} frames")

dm_commands, wfs_signal, nn_modes = wfs_rec.data
psf_frame, psf_loop_status = psf_rec.data

now = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M")

# comments = input("Comment something about the observation: ")

psf_frame=psf_frame
dm_commands=dm_commands
m2c=M2C_SHM.get_data()
WFS_PUP=WFS_PUP
loop_gain=LOOP_GAIN_SHM.get_data()[0,0]
loop_leak=LOOP_LEAK_SHM.get_data()[0,0]
wfs_fps=WFS_FPS_SHM.get_data()[0,0]
wfs_gain=WFS_GAIN_SHM.get_data()[0,0]
sci_dit=SCI_DIT_SHM.get_data()[0,0]
sci_fps=SCI_FPS_SHM.get_data()[0,0]
sci_gain=SCI_GAIN_SHM.get_data()[0,0]
# comments=comments

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
save_folder = Path(config["output"]["hdf5_dir"]) / DATE
save_folder.mkdir(parents=True, exist_ok=True)


file_name = f"Star_{now}.hdf5"
hdf5_path = str(save_folder / file_name)

with h5py.File(hdf5_path, "w") as file:
    grp_wfs = file.create_group('WFS')
    grp_wfs.attrs["Loop_Gain"] = loop_gain
    grp_wfs.attrs["Loop_Leak"] = loop_leak
    grp_wfs.attrs["Loop_Freq"] = wfs_fps

    dset_wfs = grp_wfs.create_dataset('WFS_Images', data = wfs_signal)
    dset_wfs.attrs['Gain'] = wfs_gain
    dset_wfs.attrs['FPS'] = wfs_fps

    grp_wfs.create_dataset('Valid_Pixel_Map', data = WFS_PUP)
    grp_wfs.create_dataset('DM_commands', data = dm_commands)
    grp_wfs.create_dataset('DM_TimeStamps', data = wfs_rec.timestamps)

    # grp_wfs.create_dataset('DM_offset', data = data['dmOffset'].squeeze())
    grp_wfs.create_dataset('WFS_measurements', data = nn_modes)



with h5py.File(hdf5_path, "a") as file:
    file.attrs['Is_Closed_Loop'] = True
    grp_science = file.create_group('Science')
    # grp_science.attrs['Target'] = header['TARGET_NAME']
    grp_science.attrs['Elevation'] = 90
    grp_science.attrs['InitialTimeStamp'] = psf_rec.timestamps[0]

    # if 'VMAG' in list(header.keys()):
    #     grp_science.attrs['Vmag'] = header['VMAG']
    #     grp_science.attrs['Rmag'] = header['RMAG']
    #     grp_science.attrs['Jmag'] = header['JMAG']
    #     grp_science.attrs['Hmag'] = header['HMAG']

    # else:
    #     print('***/!\\**** WRANING ****/!\\***')
    #     print('Target not found in Symbad')
    #     print('Check target name')
    #     print('***/!\\**** WRANING ****/!\\***')

    dset_science = grp_science.create_dataset('Science_PSFs', data = psf_frame)
    dset_science.attrs['Exposure_Time'] = sci_dit
    dset_science.attrs['FPS'] = sci_fps
    dset_science.attrs['Gain'] = sci_gain
    dset_science.attrs['Sampling'] = 8.
    dset_science.attrs['Wavelength'] = 635e-9


dm_modes = np.load(config["calibration"]["dm_modes"])
z_modes = np.load(config["calibration"]["z_modes"])

with h5py.File(hdf5_path, "a") as file:
    grp_calibration = file.create_group('Calibration')
    # dset_iMat = grp_calibration.create_dataset("Interaction_Matrix", data = data['s2m'])
    # dset_iMat.attrs['Wavelength'] = 500e-9
    grp_calibration.create_dataset("M2C", data = m2c)
    grp_calibration.create_dataset('DM_modes', data = dm_modes)
    grp_calibration.create_dataset('Z_full_resolution', data = z_modes)
    grp_calibration.attrs['Diameter'] = 0.6
    grp_calibration.attrs['Science_Calibration_Wavelength'] = 635e-9
    grp_calibration.attrs['AO_Calibration_Wavelength'] = 635-9
    grp_calibration.attrs['Dtelescope'] = 1
    grp_calibration.attrs['Dcalib'] = 1.
    grp_calibration.attrs['Actuator_Pitch'] = 1.
    grp_calibration.attrs['Total_Number_Of_Actuators'] = 97
    grp_calibration.attrs['Total_Number_Of_Controlled_Modes'] = 80



######## ANALYSIS ##########

p2 = PSF_Processing(hdf5_path, batch_size=50)
# p2.ProcessDark()
p2.SetSkyOrCalibContiditons()
p2.SetPSFModel()
p2.AnalyzeAllTheFile()


atm_char = Atmosphere_Characterization(hdf5_path, batch_size=1000, filter_TT=False)
atm_char.AnalyzeAllTheFile()


av = AnalysisViewer(hdf5_path)

av.CreateAtmosphericAnalysisFigures()
av.CreatePSFAnalysisFigures()


###### PSF ###########

import subprocess
from datetime import datetime

file_title = str(file_name).split("/")[-1].split(".hdf5")[0]
target_name = file_title.split("-")[0]

cmd = [
    "typst",
    "compile",
    "ao_report.typ",
    "ao_report" + file_title + ".pdf",
    "--input",
    "AOtitle=" + str(file_name).split("/")[-1].split(".hdf5")[0],
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


report_file_name = Path("ao_report" + file_title + ".pdf")

save_folder = Path(config["output"]["report_dir"]) / DATE.lstrip("/")
save_folder.mkdir(parents=True, exist_ok=True)

shutil.move(str(report_file_name), str(save_folder / report_file_name.name))