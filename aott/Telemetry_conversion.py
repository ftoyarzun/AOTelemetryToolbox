import h5py
import numpy as np
import pylab as plt
from pathlib import Path
from datetime import datetime
from astropy.io import fits


def ResetFile(file_name):
    with h5py.File(file_name, "w") as file:
        pass


def FillWFSData(file_name, data, dm_dates, saveOCAMFrames=True):
    with h5py.File(file_name, "w") as file:
        grp_wfs = file.create_group("WFS")
        grp_wfs.attrs["Loop_Gain"] = data["lpGain"][0][0]
        grp_wfs.attrs["Loop_Leak"] = data["lpLeak"][0][0]
        grp_wfs.attrs["Loop_Freq"] = data["ocamFps"]
        grp_wfs.attrs["Modulator_Frequency"] = data["modulatorFrequency"]
        grp_wfs.attrs["ModulatorRadius"] = data["modulatorRadius"]
        grp_wfs.attrs["ModulatorOn-Off"] = data["modulatorOn-Off"]

        if saveOCAMFrames:
            dset_ocam = grp_wfs.create_dataset("OCAM_Images", data=data["ocamCube"])
        else:
            dset_ocam = grp_wfs.create_dataset(
                "OCAM_Images", data=data["ocamCube"][:1000]
            )
        dset_ocam.attrs["Gain"] = data["ocamGain"]
        dset_ocam.attrs["Temp"] = data["ocamTemp"]
        dset_ocam.attrs["FPS"] = data["ocamFps"]
        dset_ocam_dark = grp_wfs.create_dataset("Dark", data=data["ocamDark"])
        dset_ocam_ref = grp_wfs.create_dataset("Reference_Frame", data=data["ref"])

        grp_wfs.create_dataset("Valid_Pixel_Map", data=data["validPixels"])
        grp_wfs.create_dataset("DM_commands", data=data["dmCLCube"].squeeze())
        # grp_wfs.create_dataset('DM_commands', data = data['dmCmdCube'].squeeze())
        grp_wfs.create_dataset("DM_TimeStamps", data=dm_dates)

        grp_wfs.create_dataset("DM_flat", data=data["dmFlat"].squeeze())
        grp_wfs.create_dataset("DM_offset", data=data["dmOffset"].squeeze())
        grp_wfs.create_dataset("WFS_measurements", data=data["modeCube"])


def FillScienceData(file_name, cube, header):
    with h5py.File(file_name, "a") as file:
        file.attrs["Is_Closed_Loop"] = "CL" in header["LOOP"]
        grp_science = file.create_group("Science")
        grp_science.attrs["Target"] = header["TARGET_NAME"]
        grp_science.attrs["Elevation"] = header["ALT"]
        grp_science.attrs["InitialTimeStamp"] = header["InitialTimeStamp"]

        if "VMAG" in list(header.keys()):
            grp_science.attrs["Vmag"] = header["VMAG"]
            grp_science.attrs["Rmag"] = header["RMAG"]
            grp_science.attrs["Jmag"] = header["JMAG"]
            grp_science.attrs["Hmag"] = header["HMAG"]

        else:
            print("***/!\\**** WRANING ****/!\\***")
            print("Target not found in Symbad")
            print("Check target name")
            print("***/!\\**** WRANING ****/!\\***")

        dset_science = grp_science.create_dataset("Science_PSFs", data=cube[:-1])
        dset_science.attrs["Exposure_Time"] = header["DIT"] * 1e-3
        dset_science.attrs["FPS"] = header["FPS"]
        dset_science.attrs["Gain"] = "High"
        dset_science.attrs["Sampling"] = 2.8
        dset_science.attrs["Wavelength"] = 1450e-9
        dset_science_dark = grp_science.create_dataset("Dark", data=cube[-1])


def FillCalibrationData(file_name, data, dm_modes, z_modes):
    with h5py.File(file_name, "a") as file:
        grp_calibration = file.create_group("Calibration")
        dset_iMat = grp_calibration.create_dataset(
            "Interaction_Matrix", data=data["s2m"]
        )
        dset_iMat.attrs["Wavelength"] = 500e-9
        grp_calibration.create_dataset("M2C", data=data["m2c"])
        grp_calibration.create_dataset("DM_modes", data=dm_modes)
        grp_calibration.create_dataset("Z_full_resolution", data=z_modes)
        grp_calibration.attrs["Diameter"] = 1.52
        grp_calibration.attrs["Science_Calibration_Wavelength"] = 1550e-9
        grp_calibration.attrs["AO_Calibration_Wavelength"] = 652 - 9
        grp_calibration.attrs["Dtelescope"] = 35.5e-3
        grp_calibration.attrs["Dcalib"] = 37.5e-3
        grp_calibration.attrs["Actuator_Pitch"] = 2.5e-3
        grp_calibration.attrs["Total_Number_Of_Actuators"] = data["m2c"].shape[0]
        grp_calibration.attrs["Total_Number_Of_Controlled_Modes"] = data["m2c"].shape[1]
