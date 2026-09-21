import h5py

import numpy as np
import pylab as plt

from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from maoppy.utils import circavg
from maoppy.instrument import papyrus

from scipy.signal import welch

from datetime import datetime


def GetSignalPSD(signal, period):
    return welch(signal, 1 / period, nperseg=500)


class AnalysisViewer:
    def __init__(self, file_name):
        with h5py.File(file_name, "r") as file:

            self.wfs_analysis = None
            self.science_analysis = None
            self.gsc_analysis = None
            if "WFS" in file:
                wfs_grp = file["WFS"]
                self.ocam_gain = wfs_grp["WFS_Images"].attrs["Gain"]

                if "Analysis" in wfs_grp:
                    self.wfs_analysis = True
                    analysis_grp = wfs_grp["Analysis"]
                    self.wfs_r0 = analysis_grp["r0"][:]
                    self.wfs_r0_units = analysis_grp["r0"].attrs["Units"]
                    self.wfs_iteration_times = analysis_grp["Iteration_Times"][:]
                    self.wfs_iteration_times = [datetime.fromtimestamp(t) for t in self.wfs_iteration_times]

                    self.wfs_tau0 = analysis_grp["tau0"][:]
                    self.wfs_tau0_units = analysis_grp["tau0"].attrs["Units"]

                    self.wfs_V0 = analysis_grp["V0"][:]
                    self.wfs_V0_zernike = analysis_grp["V0_Zernike"][:]
                    self.wfs_effective_gain = analysis_grp["Effective_Gain"][:]
                    self.wfs_effective_delay = analysis_grp["Measured_Loop_Delay"][:]
                    self.wfs_V0_units = analysis_grp["V0"].attrs["Units"]

            if "Science" in file:
                sci_grp = file["Science"]
                self.fps = sci_grp["Science_PSFs"].attrs["FPS"]

                if "Vmag" in sci_grp.attrs:
                    self.VMag = sci_grp.attrs["Vmag"]
                    self.RMag = sci_grp.attrs["Rmag"]
                    self.JMag = sci_grp.attrs["Jmag"]
                    self.HMag = sci_grp.attrs["Hmag"]
                else:
                    self.VMag = False

                self.period = 1 / self.fps
                if "Analysis" in sci_grp:
                    self.science_analysis = True
                    analysis_grp = sci_grp["Analysis"]
                    long_exp_grp = analysis_grp["Long_Exposure"]
                    self.long_exp_r0 = long_exp_grp["r0"][:]
                    self.long_exp_sr_fit = long_exp_grp["sr_fit"][:]
                    self.long_exp_sr_otf = long_exp_grp["sr_otf"][:]
                    self.long_exp_psf_model = long_exp_grp["psf_model"][:]
                    self.long_exp_psf_stack = long_exp_grp["psf_stack"][:]
                    self.long_exp_iteration_times = long_exp_grp["Iteration_Times"][:]
                    self.long_exp_iteration_times = [datetime.fromtimestamp(t) for t in self.long_exp_iteration_times]

                    short_exp_grp = analysis_grp["Short_Exposure"]
                    self.short_exp_CoG_X = short_exp_grp["CoG-X"][:]
                    self.short_exp_CoG_Y = short_exp_grp["CoG-Y"][:]
                    self.jitter = short_exp_grp["Jitter"][:]

                    self.psf_wavelength = sci_grp["Science_PSFs"].attrs["Wavelength"]
                    self.psf_sampling = sci_grp["Science_PSFs"].attrs["Sampling"]
                    self.diameter = file["Calibration"].attrs["Diameter"]

    def MakeR0Plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_r0,
            label="r0 from AO telemetry",
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            self.long_exp_iteration_times[:-1],
            self.long_exp_r0 * 100,
            label="r0 from PSF processing",
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"$r_0$ @ 500 nm ({self.wfs_r0_units})")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 15)
        ax.legend()

        fig_path = "AtmosphereAnalysis_r0.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def MakeTau0Plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_tau0,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"$\\tau_0$ @ 500 nm ({self.wfs_tau0_units})")
        ax.set_xlabel("Time (s)")

        fig_path = "AtmosphereAnalysis_tau0.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def MakeV0Plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_V0,
            label="V0 from structure function",
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_V0_zernike,
            label="V0 from Zernike statistics",
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"$V_0$ @ 500 nm ({self.wfs_V0_units})")
        ax.set_xlabel("Time(s)")
        ax.legend()

        fig_path = "AtmosphereAnalysis_V0.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def MakeLoopParamPlots(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_effective_gain,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"Measured loop gain")
        ax.set_xlabel("Time(s)")

        fig_path = "AtmosphereAnalysis_loop_gain.png"
        fig.savefig(fig_path, bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            self.wfs_iteration_times[:-1],
            self.wfs_effective_delay,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"Measured loop delay (frames)")
        ax.set_xlabel("Time(s)")

        fig_path = "AtmosphereAnalysis_loop_delay.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def CreateAtmosphericAnalysisFigures(self):
        self.MakeR0Plot()
        self.MakeTau0Plot()
        self.MakeV0Plot()
        self.MakeLoopParamPlots()

    def MakeSRPlot(self):
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            self.long_exp_iteration_times[:-1],
            self.long_exp_sr_fit,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel(f"Strehl ratio @ {self.psf_wavelength * 1e9:.0f} nm")
        ax.set_xlabel("Time (s)")

        fig_path = "PSFAnalysis.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def MakePSFImage(self):
        def setplt(tab):
            rad2arcsec = 206265
            norm = LogNorm(vmin=1e-3, vmax=1)
            cmap = "Spectral_r"
            pix_scale = self.psf_wavelength / self.diameter / self.psf_sampling
            nx = tab.shape[0]
            axis = np.linspace(-nx // 2, nx // 2, nx) * pix_scale * rad2arcsec
            im1 = plt.imshow(
                tab / tab.max(),
                norm=norm,
                cmap=cmap,
                extent=[axis[0], axis[-1], axis[0], axis[-1]],
            )
            plt.colorbar(im1, fraction=0.046, pad=0.04)
            corr_zone = Circle(
                [0, 0],
                self.psf_wavelength / self.diameter * papyrus.nact / 2 * rad2arcsec,
                fc="none",
                ec="k",
                ls=":",
            )
            plt.gca().add_artist(corr_zone)
            plt.xlabel("[arcsec]")
            plt.ylabel("[arcsec]")
            hfov = 2
            plt.xlim(-hfov, hfov)
            plt.ylim(-hfov, hfov)

        fig2, ax = plt.subplots(2, 2, figsize=(9, 8))
        plt.clf()

        index_min_sr = np.where(self.long_exp_sr_fit == self.long_exp_sr_fit.min())[0]
        index_max_sr = np.where(self.long_exp_sr_fit == self.long_exp_sr_fit.max())[0]

        # plt.suptitle('Strehl = %.1f %%\nSeeing = %.1f\"'%(100*sr, seeing))
        plt.subplot(221)
        plt.title("data - min SR")
        setplt(self.long_exp_psf_stack[index_min_sr].squeeze())

        plt.subplot(222)
        plt.title("fit")
        setplt(self.long_exp_psf_model[index_min_sr].squeeze())

        plt.subplot(223)
        plt.title("data - max SR")
        setplt(self.long_exp_psf_stack[index_max_sr].squeeze())

        plt.subplot(224)
        plt.title("fit")
        setplt(self.long_exp_psf_model[index_max_sr].squeeze())

        plt.tight_layout()

        fig_path = "PSFFrames.png"
        fig2.savefig(fig_path, bbox_inches="tight")

    def CreatePSFAnalysisFigures(self):
        self.MakeSRPlot()
        self.MakePSFImage()
        self.MakeJitterPlot()
        self.MakeCoGStatsPlots()

    def MakeJitterPlot(self):
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            self.long_exp_iteration_times[:-1],
            self.jitter,
            linestyle="--",
            marker="o",
            linewidth=2,
        )
        ax.set_ylabel("Jitter ($\\lambda/D$)")
        ax.set_xlabel("Time (s)")
        ax.legend(["x-cog", "y-cog"])
        fig_path = "PSFJitter.png"
        fig.savefig(fig_path, bbox_inches="tight")

    def MakeCoGStatsPlots(self):

        _, self.x_psd = GetSignalPSD(self.short_exp_CoG_X, self.period)
        self.f, self.y_psd = GetSignalPSD(self.short_exp_CoG_Y, self.period)

        df = self.f[1] - self.f[0]

        cum_x = np.cumsum(self.x_psd) * df
        cum_y = np.cumsum(self.y_psd) * df

        var_x = np.var(self.short_exp_CoG_X)
        var_y = np.var(self.short_exp_CoG_Y)

        cum_x = cum_x / cum_x[-1] * var_x
        cum_y = cum_y / cum_y[-1] * var_y

        fig, ax = plt.subplots()
        ax.loglog(self.f, self.x_psd, label="x-cog")
        ax.loglog(self.f, self.y_psd, label="y-cog")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD ( $(\\lambda/D)^2/Hz$ )")
        ax.legend()
        fig_path = "CoG_PSD.png"
        fig.savefig(fig_path, bbox_inches="tight")

        fig, ax = plt.subplots()
        ax.semilogx(self.f, np.sqrt(cum_x), label="x-cog")
        ax.semilogx(self.f, np.sqrt(cum_y), label="y-cog")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cumulative jitter $(\\lambda/D)$")
        ax.legend()
        ax.set_xlim(self.f[1], self.f[-1])
        fig_path = "Cumulative_Jitter.png"
        fig.savefig(fig_path, bbox_inches="tight")
