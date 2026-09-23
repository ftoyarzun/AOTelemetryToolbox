import h5py
import json
import os

import numpy as np
import pylab as plt
import matplotlib.dates as mdates

from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Circle
from maoppy.utils import circavg
from maoppy.instrument import Instrument

from scipy.signal import welch

from datetime import datetime

from aott.atmosphere_characterization_tools import find_status_runs
from aott.frozen_flow_profiler import plot_correlation, plot_layer_maps, plot_layer_profile, read_first_batch


def GetSignalPSD(signal, period):
    return welch(signal, 1 / period, nperseg=500)


# PNG filename(s) per figures-manifest key: the whitelist RemoveFigureFiles
# deletes from, so cleanup never touches an unrelated PNG such as a logo.
_FIGURE_FILES = {
    "r0": ["AtmosphereAnalysis_r0.png"],
    "L0": ["AtmosphereAnalysis_L0.png"],
    "tau0": ["AtmosphereAnalysis_tau0.png"],
    "V0": ["AtmosphereAnalysis_V0.png"],
    "loop_params": ["AtmosphereAnalysis_loop_gain.png", "AtmosphereAnalysis_loop_delay.png"],
    "psd_comparison": ["AtmosphereAnalysis_PSD_Comparison.png"],
    "loop_bandwidth": ["AtmosphereAnalysis_LoopBandwidth.png"],
    "sr": ["PSFAnalysis.png"],
    "open_loop_seeing": ["PSFAnalysis_OpenLoopSeeing.png"],
    "psf_frames": ["PSFFrames.png"],
    "psf_frames_openloop": ["PSFFrames_OpenLoop.png"],
    "jitter": ["PSFJitter.png"],
    "cog_stats": ["CoG_PSD.png", "Cumulative_Jitter.png"],
    "frozen_flow": ["correlation.png", "layer_maps.png", "layers.png"],
}


def _has_data(x):
    """True if x is a non-None array/list with at least one entry."""
    return x is not None and len(x) > 0


def _split_at_gaps(times_raw, gap_factor=3.0):
    """
    Contiguous-run boundaries of a 1D time array, as a list of (start, end)
    index pairs, splitting wherever a gap between consecutive samples exceeds
    `gap_factor` times the typical (median) gap: an open-loop stretch, or a
    run too short to analyze, was skipped there.
    """
    times_raw = np.asarray(times_raw)
    if len(times_raw) < 2:
        return [(0, len(times_raw))]
    diffs = np.diff(times_raw)
    typical = np.median(diffs)
    breaks = np.where(diffs > gap_factor * typical)[0] + 1 if typical > 0 else np.array([], dtype=int)
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(times_raw)]))
    return list(zip(starts.tolist(), ends.tolist()))


def _format_time_axis(ax):
    """Label a datetime x-axis as HH:MM:SS, whatever the tick spacing."""
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)
    for scale in (1 / mdates.HOURS_PER_DAY, 1 / mdates.MINUTES_PER_DAY, 1 / mdates.SEC_PER_DAY):
        formatter.scaled[scale] = "%H:%M:%S"
    # Sub-second ticks (very short series): keep hh:mm:ss, plus one decimal
    formatter.scaled[1 / mdates.MUSECONDS_PER_DAY] = (
        lambda x, pos=None: mdates.num2date(x).strftime("%H:%M:%S.%f")[:-5])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel("Time")


class AnalysisViewer:
    def __init__(self, file_name):
        # Defaults, so a missing group or regime just skips its plots
        self.wfs_analysis = None
        self.science_analysis = None
        self.gsc_analysis = None

        self.wfs_r0 = None
        self.wfs_r0_units = None
        self.wfs_iteration_times = None
        self.wfs_L0 = None
        self.wfs_L0_units = None
        self.wfs_tau0 = None
        self.wfs_tau0_units = None
        self.wfs_tau0_autocorrelation = None
        self.wfs_V0 = None
        self.wfs_V0_autocorrelation = None
        self.wfs_effective_gain = None
        self.wfs_effective_delay = None
        self.wfs_V0_units = None
        self.wfs_psd_comparison = None
        self.wfs_loop_bandwidth = None
        self.wfs_open_loop_psd = None
        # Frozen-flow profiler: per-batch V0/tau0/r0, and the first batch's fit for its figures
        self.frozen_flow = None
        self.frozen_flow_first_batch = None

        self.long_exp_r0 = None
        self.long_exp_sr_fit = None
        self.long_exp_sr_otf = None
        self.long_exp_psf_model = None
        self.long_exp_psf_stack = None
        self.long_exp_iteration_times = None
        self.long_exp_iteration_times_raw = None
        self.open_loop_r0 = None
        self.open_loop_psf_model = None
        self.open_loop_psf_stack = None
        self.open_loop_iteration_times = None
        self.open_loop_iteration_times_raw = None
        self.short_exp_CoG_X = None
        self.short_exp_CoG_Y = None
        self.jitter = None
        self.jitter_iteration_times = None
        self.jitter_iteration_times_raw = None
        self.open_loop_jitter = None
        self.short_exp_is_closed_loop = None
        self.short_exp_transition_buffer = 0
        self.psf_wavelength = None
        self.psf_sampling = None
        self.diameter = None

        # Which figures actually got written, and small mean/std summaries of
        # the quantity each one plots -- dumped to JSON by SaveFigureManifest
        # so ao_report.typ can include exactly the sections this file has
        # data for, and print a one-line summary number next to each figure,
        # without ever referencing a PNG that was never produced.
        self.manifest = {"figures": {}, "stats": {}}

        with h5py.File(file_name, "r") as file:

            if "WFS" in file:
                wfs_grp = file["WFS"]

                if "Analysis" in wfs_grp and "r0" in wfs_grp["Analysis"]:
                    self.wfs_analysis = True
                    analysis_grp = wfs_grp["Analysis"]
                    self.wfs_r0 = analysis_grp["r0"][:]
                    self.wfs_r0_units = analysis_grp["r0"].attrs["Units"]
                    self.wfs_iteration_times_raw = analysis_grp["Iteration_Times"][:]
                    self.wfs_iteration_times = [datetime.fromtimestamp(t) for t in self.wfs_iteration_times_raw]

                    self.wfs_L0 = analysis_grp["L0"][:]
                    self.wfs_L0_units = analysis_grp["L0"].attrs["Units"]

                    self.wfs_tau0 = analysis_grp["tau0"][:]
                    self.wfs_tau0_units = analysis_grp["tau0"].attrs["Units"]
                    self.wfs_tau0_autocorrelation = analysis_grp["tau0_Autocorrelation"][:]

                    self.wfs_V0 = analysis_grp["V0"][:]
                    self.wfs_V0_autocorrelation = analysis_grp["V0_Autocorrelation"][:]
                    self.wfs_effective_gain = analysis_grp["Effective_Gain"][:]
                    self.wfs_effective_delay = analysis_grp["Measured_Loop_Delay"][:]
                    self.wfs_V0_units = analysis_grp["V0"].attrs["Units"]

                    self.wfs_psd_comparison = None
                    if "PSD_Comparison" in analysis_grp:
                        psd_grp = analysis_grp["PSD_Comparison"]
                        self.wfs_psd_comparison = dict(
                            modes=psd_grp["Modes"][:],
                            frequency=psd_grp["Frequency"][:],
                            dm_psd=psd_grp["DM_PSD"][:],
                            wfs_psd=psd_grp["WFS_PSD"][:],
                            iteration_times=psd_grp["Iteration_Times"][:],
                        )

                    self.wfs_loop_bandwidth = None
                    if "Loop_Bandwidth" in analysis_grp:
                        bw_grp = analysis_grp["Loop_Bandwidth"]
                        self.wfs_loop_bandwidth = dict(
                            radial_orders=bw_grp["Radial_Orders"][:],
                            crossover_frequency=bw_grp["Crossover_Frequency"][:],
                            iteration_times=bw_grp["Iteration_Times"][:],
                        )

                    self.wfs_open_loop_psd = None
                    if "Open_Loop_PSD" in analysis_grp:
                        ol_grp = analysis_grp["Open_Loop_PSD"]
                        self.wfs_open_loop_psd = dict(
                            modes=ol_grp["Modes"][:],
                            frequency=ol_grp["Frequency"][:],
                            psd=ol_grp["PSD"][:],
                            iteration_times=ol_grp["Iteration_Times"][:],
                        )

                if "Analysis" in wfs_grp and "Frozen_Flow" in wfs_grp["Analysis"]:
                    ff_grp = wfs_grp["Analysis"]["Frozen_Flow"]
                    times_raw = ff_grp["Iteration_Times"][:]
                    self.frozen_flow = dict(
                        iteration_times_raw=times_raw,
                        iteration_times=[datetime.fromtimestamp(t) for t in times_raw],
                        V0=ff_grp["V0"][:],
                        tau0=ff_grp["tau0"][:],
                        r0=ff_grp["r0"][:],
                        n_layers=ff_grp["N_Layers"][:],
                    )
                    if "First_Batch" in ff_grp:
                        self.frozen_flow_first_batch = read_first_batch(ff_grp)

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

                    if "Long_Exposure" in analysis_grp:
                        long_exp_grp = analysis_grp["Long_Exposure"]
                        if long_exp_grp["r0"].shape[0] > 0:
                            self.long_exp_r0 = long_exp_grp["r0"][:]
                            self.long_exp_sr_fit = long_exp_grp["sr_fit"][:]
                            self.long_exp_sr_otf = long_exp_grp["sr_otf"][:]
                            self.long_exp_psf_model = long_exp_grp["psf_model"][:]
                            self.long_exp_psf_stack = long_exp_grp["psf_stack"][:]
                            self.long_exp_iteration_times_raw = long_exp_grp["Iteration_Times"][:]
                            self.long_exp_iteration_times = [
                                datetime.fromtimestamp(t) for t in self.long_exp_iteration_times_raw
                            ]

                    if "Long_Exposure_OpenLoop" in analysis_grp:
                        open_loop_grp = analysis_grp["Long_Exposure_OpenLoop"]
                        if open_loop_grp["r0"].shape[0] > 0:
                            self.open_loop_r0 = open_loop_grp["r0"][:]
                            self.open_loop_psf_model = open_loop_grp["psf_model"][:]
                            self.open_loop_psf_stack = open_loop_grp["psf_stack"][:]
                            self.open_loop_iteration_times_raw = open_loop_grp["Iteration_Times"][:]
                            self.open_loop_iteration_times = [
                                datetime.fromtimestamp(t) for t in self.open_loop_iteration_times_raw
                            ]

                    if "Short_Exposure" in analysis_grp:
                        short_exp_grp = analysis_grp["Short_Exposure"]
                        self.short_exp_CoG_X = short_exp_grp["CoG-X"][:]
                        self.short_exp_CoG_Y = short_exp_grp["CoG-Y"][:]
                        if "Is_Closed_Loop" in short_exp_grp:
                            self.short_exp_is_closed_loop = short_exp_grp["Is_Closed_Loop"][:]
                        # Frames PSF_Processing skipped after each loop transition
                        # (absent, i.e. 0, in files analysed before the buffer existed)
                        self.short_exp_transition_buffer = int(short_exp_grp.attrs.get("Transition_Buffer_Frames", 0))
                        # Jitter is one entry per PSF batch, same cadence and
                        # order as Long_Exposure/Long_Exposure_OpenLoop, so it
                        # reuses their Iteration_Times rather than its own.
                        if "Jitter" in short_exp_grp and short_exp_grp["Jitter"].shape[0] > 0:
                            self.jitter = short_exp_grp["Jitter"][:]
                            self.jitter_iteration_times = self.long_exp_iteration_times
                            self.jitter_iteration_times_raw = self.long_exp_iteration_times_raw
                        if "Jitter_OpenLoop" in short_exp_grp and short_exp_grp["Jitter_OpenLoop"].shape[0] > 0:
                            self.open_loop_jitter = short_exp_grp["Jitter_OpenLoop"][:]

                    self.psf_wavelength = sci_grp["Science_PSFs"].attrs["Wavelength"]
                    self.psf_sampling = sci_grp["Science_PSFs"].attrs["Sampling"]
                    calibration_grp = file["Calibration"]
                    self.diameter = calibration_grp.attrs["Diameter"]
                    self.Obstruction_ratio = calibration_grp.attrs['Obstruction_ratio']
                    self.actuators_in_dm_diameter = calibration_grp.attrs['Actuators_in_diameter']
                    self.total_number_of_actuators = calibration_grp.attrs['Total_Number_Of_Actuators']
                    self.total_number_of_controlled_modes = calibration_grp.attrs['Total_Number_Of_Controlled_Modes']
                    self.SkyCalibPupilRatio = calibration_grp.attrs['SkyCalibPupilRatio']

            self.instrument = Instrument(D=self.diameter, 
                                         occ=self.Obstruction_ratio,
                                         res = self.psf_wavelength / self.diameter / self.psf_sampling)

            nb_act_lin = self.actuators_in_dm_diameter
            self.instrument.nact = round(nb_act_lin * self.SkyCalibPupilRatio * np.sqrt(self.total_number_of_controlled_modes/self.total_number_of_actuators))

    def _plot_segmented(self, ax, raw_times, iteration_times, values, label=None, **plot_kwargs):
        """
        Plot a time series as one line per contiguous run (detected from gaps
        in `raw_times`), so no line is drawn across a gap in the analysis.
        `label` is attached to the first run only.
        """
        values = np.asarray(values)
        for i, (start, end) in enumerate(_split_at_gaps(raw_times)):
            ax.plot(
                iteration_times[start:end],
                values[start:end],
                label=label if i == 0 else None,
                **plot_kwargs,
            )

    def _flag_figure(self, key):
        """Mark a figure as produced, for the JSON manifest ao_report.typ
        reads to decide which sections to include."""
        self.manifest["figures"][key] = True

    def _record_stat(self, key, values):
        """Store mean/std of a per-batch quantity in the manifest, printed as
        a small summary number next to that figure's caption in the report."""
        values = np.asarray(values, dtype=float)
        self.manifest["stats"][f"{key}_mean"] = float(np.nanmean(values))
        self.manifest["stats"][f"{key}_std"] = float(np.nanstd(values))

    def SaveFigureManifest(self, path="report_data.json"):
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def RemoveFigureFiles(self):
        """Delete every PNG this run actually wrote (per self.manifest), for
        after ao_report.typ has compiled -- Typst embeds each image directly
        in the PDF, so the standalone PNGs left in the working directory are
        pure clutter at that point."""
        for key, produced in self.manifest["figures"].items():
            if not produced:
                continue
            for fname in _FIGURE_FILES.get(key, []):
                if os.path.exists(fname):
                    os.remove(fname)

    def MakeR0Plot(self):
        if (not _has_data(self.wfs_r0) and not _has_data(self.long_exp_r0) and not _has_data(self.open_loop_r0)
                and self.frozen_flow is None):
            return
        fig, ax = plt.subplots(figsize=(6, 4))

        if _has_data(self.wfs_r0):
            self._plot_segmented(
                ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_r0,
                label="r0 from AO telemetry", linestyle="--", marker="o", linewidth=2, color="C0",
            )
            self._record_stat("r0_wfs", self.wfs_r0)
        # PSF-derived r0 is one color regardless of loop status, and closed-
        # vs open-loop are distinguished by marker only, so the two colors on
        # this plot always mean "which estimator" (telemetry vs PSF), never
        # "which loop status".
        if _has_data(self.long_exp_r0):
            self._plot_segmented(
                ax, self.long_exp_iteration_times_raw, self.long_exp_iteration_times, self.long_exp_r0 * 100,
                label="r0 from PSF processing (closed loop)",
                linestyle="--", marker="o", linewidth=2, color="C1",
            )
            self._record_stat("r0_psf_closed", self.long_exp_r0 * 100)
        if _has_data(self.open_loop_r0):
            self._plot_segmented(
                ax, self.open_loop_iteration_times_raw, self.open_loop_iteration_times, self.open_loop_r0 * 100,
                label="r0 from PSF processing (open loop)",
                linestyle="--", marker="s", linewidth=2, color="C1",
            )
            self._record_stat("r0_psf_open", self.open_loop_r0 * 100)
        if self.frozen_flow is not None:
            self._plot_segmented(
                ax, self.frozen_flow["iteration_times_raw"], self.frozen_flow["iteration_times"],
                self.frozen_flow["r0"], label="r0 from frozen-flow profiler",
                linestyle="--", marker="o", linewidth=2, color="C2",
            )
            self._record_stat("r0_frozen_flow", self.frozen_flow["r0"])
        ax.set_ylabel(f"$r_0$ @ 500 nm ({self.wfs_r0_units if self.wfs_r0_units else 'cm'})")
        _format_time_axis(ax)
        ax.legend()

        fig_path = "AtmosphereAnalysis_r0.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("r0")

    def MakeL0Plot(self):
        if not _has_data(self.wfs_L0):
            return
        fig, ax = plt.subplots(figsize=(6, 4))

        self._plot_segmented(
            ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_L0,
            linestyle="--", marker="o", linewidth=2, color="C0",
        )
        ax.set_ylabel(f"$L_0$ ({self.wfs_L0_units})")
        _format_time_axis(ax)

        fig_path = "AtmosphereAnalysis_L0.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("L0")
        self._record_stat("L0", self.wfs_L0)

    def MakeTau0Plot(self):
        if not _has_data(self.wfs_tau0) and self.frozen_flow is None:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        if _has_data(self.wfs_tau0):
            self._plot_segmented(
                ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_tau0,
                label="tau0 from structure function", linestyle="--", marker="o", linewidth=2, color="C0",
            )
            self._plot_segmented(
                ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_tau0_autocorrelation,
                label="tau0 from autocorrelation", linestyle="--", marker="o", linewidth=2, color="C1",
            )
            self._record_stat("tau0", self.wfs_tau0)
            self._record_stat("tau0_autocorrelation", self.wfs_tau0_autocorrelation)
        if self.frozen_flow is not None:
            self._plot_segmented(
                ax, self.frozen_flow["iteration_times_raw"], self.frozen_flow["iteration_times"],
                self.frozen_flow["tau0"], label="tau0 from frozen-flow profiler",
                linestyle="--", marker="o", linewidth=2, color="C2",
            )
            self._record_stat("tau0_frozen_flow", self.frozen_flow["tau0"])
        ax.set_ylabel(f"$\\tau_0$ @ 500 nm ({self.wfs_tau0_units if self.wfs_tau0_units else 'ms'})")
        _format_time_axis(ax)
        ax.legend()

        fig_path = "AtmosphereAnalysis_tau0.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("tau0")

    def MakeV0Plot(self):
        if not _has_data(self.wfs_V0) and self.frozen_flow is None:
            return
        fig, ax = plt.subplots(figsize=(6, 4))

        if _has_data(self.wfs_V0):
            self._plot_segmented(
                ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_V0,
                label="V0 from structure function", linestyle="--", marker="o", linewidth=2, color="C0",
            )
            self._plot_segmented(
                ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_V0_autocorrelation,
                label="V0 from autocorrelation", linestyle="--", marker="o", linewidth=2, color="C1",
            )
            self._record_stat("V0", self.wfs_V0)
            self._record_stat("V0_autocorrelation", self.wfs_V0_autocorrelation)
        if self.frozen_flow is not None:
            self._plot_segmented(
                ax, self.frozen_flow["iteration_times_raw"], self.frozen_flow["iteration_times"],
                self.frozen_flow["V0"], label="V0 from frozen-flow profiler",
                linestyle="--", marker="o", linewidth=2, color="C2",
            )
            self._record_stat("V0_frozen_flow", self.frozen_flow["V0"])
        ax.set_ylabel(f"$V_0$ @ 500 nm ({self.wfs_V0_units if self.wfs_V0_units else 'm/s'})")
        _format_time_axis(ax)
        ax.legend()

        fig_path = "AtmosphereAnalysis_V0.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("V0")

    def MakeLoopParamPlots(self):
        if not _has_data(self.wfs_effective_gain):
            return
        fig, ax = plt.subplots(figsize=(6, 4))

        self._plot_segmented(
            ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_effective_gain,
            linestyle="--", marker="o", linewidth=2, color="C0",
        )
        ax.set_ylabel(f"Measured loop gain")
        _format_time_axis(ax)

        fig_path = "AtmosphereAnalysis_loop_gain.png"
        fig.savefig(fig_path, bbox_inches="tight")

        fig, ax = plt.subplots(figsize=(6, 4))

        self._plot_segmented(
            ax, self.wfs_iteration_times_raw, self.wfs_iteration_times, self.wfs_effective_delay,
            linestyle="--", marker="o", linewidth=2, color="C0",
        )
        ax.set_ylabel(f"Measured loop delay (frames)")
        _format_time_axis(ax)

        fig_path = "AtmosphereAnalysis_loop_delay.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("loop_params")
        self._record_stat("loop_gain", self.wfs_effective_gain)
        self._record_stat("loop_delay", self.wfs_effective_delay)

    def MakePSDComparisonPlot(self):
        if not self.wfs_psd_comparison and not self.wfs_open_loop_psd:
            return
        # Both use the same requested mode list (Atmosphere_Characterization's
        # psd_comparison_modes), so they can share one set of panels even when
        # only one of the two is available.
        modes = self.wfs_psd_comparison["modes"] if self.wfs_psd_comparison else self.wfs_open_loop_psd["modes"]

        fig, axes = plt.subplots(1, len(modes), figsize=(4 * len(modes), 4), sharey=True)
        axes = np.atleast_1d(axes)

        if self.wfs_psd_comparison:
            # Most recent closed-loop batch -- one DM-vs-WFS PSD snapshot per mode.
            f = self.wfs_psd_comparison["frequency"]
            dm_psd = self.wfs_psd_comparison["dm_psd"][-1]
            wfs_psd = self.wfs_psd_comparison["wfs_psd"][-1]
            for i in range(len(modes)):
                axes[i].loglog(f, dm_psd[i], label="DM-derived (closed loop, open-loop estimate)", color="C0")
                axes[i].loglog(f, wfs_psd[i], label="WFS-derived (closed loop, residual)", color="C1")

        if self.wfs_open_loop_psd:
            # Most recent open-loop batch -- the WFS directly measures the
            # atmosphere here (no DM correction to compare against).
            f_open = self.wfs_open_loop_psd["frequency"]
            psd_open = self.wfs_open_loop_psd["psd"][-1]
            for i in range(len(modes)):
                axes[i].loglog(f_open, psd_open[i], label="WFS-derived (open loop)", color="C2", linestyle="--")

        for i, mode in enumerate(modes):
            axes[i].set_title(f"mode {mode}")
            axes[i].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("PSD (rad$^2$/Hz)")
        axes[0].legend(fontsize=8)

        fig_path = "AtmosphereAnalysis_PSD_Comparison.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("psd_comparison")

    def MakeLoopBandwidthPlot(self):
        if not self.wfs_loop_bandwidth:
            return
        orders = self.wfs_loop_bandwidth["radial_orders"]
        crossover = self.wfs_loop_bandwidth["crossover_frequency"]

        # Averaged over every closed-loop batch (time), per radial order --
        # one point per order instead of one line per order over time.
        mean_crossover = np.nanmean(crossover, axis=0)
        std_crossover = np.nanstd(crossover, axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            orders, mean_crossover, yerr=std_crossover,
            linestyle="--", marker="o", linewidth=2, color="C0", capsize=3,
        )
        ax.set_ylabel("Loop bandwidth (Hz)")
        ax.set_xlabel("Radial order n")

        fig_path = "AtmosphereAnalysis_LoopBandwidth.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("loop_bandwidth")
        self._record_stat("loop_bandwidth", crossover)

    def MakeFrozenFlowPlots(self):
        """The frozen-flow profiler's figures for its first closed-loop batch:
        data, model and residual of the correlation cube at a few lags, the
        layer maps, and C_n^2, speed and direction per layer."""
        if self.frozen_flow_first_batch is None:
            return
        fit, speed, direction = self.frozen_flow_first_batch
        plot_correlation(fit, "correlation.png")
        plot_layer_maps(fit, speed, "layer_maps.png")
        plot_layer_profile(fit.cn2, speed, direction, "layers.png")
        self._flag_figure("frozen_flow")
        self._record_stat("frozen_flow_layers", self.frozen_flow["n_layers"])

    def CreateAtmosphericAnalysisFigures(self):
        self.MakeR0Plot()
        self.MakeL0Plot()
        self.MakeTau0Plot()
        self.MakeV0Plot()
        self.MakeLoopParamPlots()
        self.MakePSDComparisonPlot()
        self.MakeLoopBandwidthPlot()
        self.MakeFrozenFlowPlots()

    def MakeSRPlot(self):
        if not _has_data(self.long_exp_sr_fit):
            return
        fig, ax = plt.subplots(figsize=(10, 4))

        self._plot_segmented(
            ax, self.long_exp_iteration_times_raw, self.long_exp_iteration_times, self.long_exp_sr_fit,
            linestyle="--", marker="o", linewidth=2, color="C0",
        )
        ax.set_ylabel(f"Strehl ratio @ {self.psf_wavelength * 1e9:.0f} nm")
        _format_time_axis(ax)

        fig_path = "PSFAnalysis.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("sr")
        self._record_stat("sr", self.long_exp_sr_fit)

    def MakeOpenLoopSeeingPlot(self):
        if not _has_data(self.open_loop_r0):
            return
        fig, ax = plt.subplots(figsize=(10, 4))

        self._plot_segmented(
            ax, self.open_loop_iteration_times_raw, self.open_loop_iteration_times, self.open_loop_r0 * 100,
            linestyle="--", marker="o", linewidth=2, color="C0",
        )
        ax.set_ylabel(f"$r_0$ @ 500 nm (cm), open loop")
        _format_time_axis(ax)

        fig_path = "PSFAnalysis_OpenLoopSeeing.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("open_loop_seeing")
        self._record_stat("open_loop_seeing_r0", self.open_loop_r0 * 100)

    def _setplt(self, tab, norm_by=None, cmap="Spectral_r", norm=None):
        rad2arcsec = 206265
        if norm is None:
            norm = LogNorm(vmin=1e-3, vmax=1)
        if norm_by is None:
            norm_by = tab.max()
        pix_scale = self.psf_wavelength / self.diameter / self.psf_sampling
        nx = tab.shape[0]
        axis = np.linspace(-nx // 2, nx // 2, nx) * pix_scale * rad2arcsec
        im1 = plt.imshow(
            tab / norm_by,
            norm=norm,
            cmap=cmap,
            extent=[axis[0], axis[-1], axis[0], axis[-1]],
        )
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        corr_zone = Circle(
            [0, 0],
            self.psf_wavelength / self.diameter * self.instrument.nact / 2 * rad2arcsec,
            fc="none",
            ec="k",
            ls=":",
        )
        plt.gca().add_artist(corr_zone)
        plt.xlabel("[arcsec]")
        plt.ylabel("[arcsec]")
        hfov = min(2, axis.max())
        plt.xlim(-hfov, hfov)
        plt.ylim(-hfov, hfov)

    def _make_psf_extremes_figure(self, psf_stack, psf_model, rank_by, fig_path, row_labels):
        """
        Shared by MakePSFImage/MakePSFImageOpenLoop: a data/fit/(fit-data) 2x3
        grid for the frames with the lowest and highest `rank_by` value
        (Strehl for closed loop, r0 for open loop -- whichever quantity that
        regime actually has). The difference panel matches
        PSF_Processing.FitPSFModel's own display=True path: same colormap/
        norm, and normalized by that row's data peak (not its own), so all
        three panels in a row share one color scale.
        """
        index_min = np.where(rank_by == rank_by.min())[0]
        index_max = np.where(rank_by == rank_by.max())[0]

        fig2, ax = plt.subplots(2, 3, figsize=(13, 8))
        plt.clf()

        for row, (index, label) in enumerate([(index_min, row_labels[0]), (index_max, row_labels[1])]):
            data = psf_stack[index].squeeze()
            fit = psf_model[index].squeeze()
            data_peak = data.max()

            plt.subplot(2, 3, row * 3 + 1)
            plt.title(f"data - {label}")
            self._setplt(data, norm_by=data_peak)

            plt.subplot(2, 3, row * 3 + 2)
            plt.title("fit")
            self._setplt(fit, norm_by=data_peak)

            plt.subplot(2, 3, row * 3 + 3)
            plt.title("fit - data")
            self._setplt(fit - data, norm_by=data_peak, cmap="RdBu", norm=SymLogNorm(1e-3, vmin=-1, vmax=1))

        plt.tight_layout()
        fig2.savefig(fig_path, bbox_inches="tight")

    def MakePSFImage(self):
        if not _has_data(self.long_exp_sr_fit):
            return
        # Best/worst is ranked by Strehl -- only meaningful for closed-loop
        # (AO-corrected) frames, so this method only ever looks at
        # long_exp_*, never open_loop_*.
        self._make_psf_extremes_figure(
            self.long_exp_psf_stack, self.long_exp_psf_model, self.long_exp_sr_fit,
            "PSFFrames.png", row_labels=["min SR", "max SR"],
        )
        self._flag_figure("psf_frames")
        self.manifest["stats"]["sr_min"] = float(np.nanmin(self.long_exp_sr_fit))
        self.manifest["stats"]["sr_max"] = float(np.nanmax(self.long_exp_sr_fit))

    def MakePSFImageOpenLoop(self):
        if not _has_data(self.open_loop_r0):
            return
        # No Strehl in open loop -- rank by r0 instead (larger r0 = better
        # seeing = "best", smaller r0 = "worst").
        self._make_psf_extremes_figure(
            self.open_loop_psf_stack, self.open_loop_psf_model, self.open_loop_r0,
            "PSFFrames_OpenLoop.png", row_labels=["worst r0", "best r0"],
        )
        self._flag_figure("psf_frames_openloop")
        self.manifest["stats"]["open_loop_r0_frames_min"] = float(np.nanmin(self.open_loop_r0) * 100)
        self.manifest["stats"]["open_loop_r0_frames_max"] = float(np.nanmax(self.open_loop_r0) * 100)

    def CreatePSFAnalysisFigures(self):
        self.MakeSRPlot()
        self.MakeOpenLoopSeeingPlot()
        self.MakePSFImage()
        self.MakePSFImageOpenLoop()
        self.MakeJitterPlot()
        self.MakeCoGStatsPlots()

    def MakeJitterPlot(self, log_ratio=10):
        if not _has_data(self.jitter) and not _has_data(self.open_loop_jitter):
            return
        fig, ax = plt.subplots(figsize=(10, 4))

        # Color = axis (x vs y), marker = loop status -- same convention as
        # MakeR0Plot, so closed/open is always a marker distinction here.
        if _has_data(self.jitter):
            self._plot_segmented(ax, self.jitter_iteration_times_raw, self.jitter_iteration_times, self.jitter[:, 0],
                    label="x-cog (closed loop)", linestyle="--", marker="o", linewidth=2, color="C0")
            self._plot_segmented(ax, self.jitter_iteration_times_raw, self.jitter_iteration_times, self.jitter[:, 1],
                    label="y-cog (closed loop)", linestyle="--", marker="o", linewidth=2, color="C1")
            self._record_stat("jitter_x_closed", self.jitter[:, 0])
            self._record_stat("jitter_y_closed", self.jitter[:, 1])
        if _has_data(self.open_loop_jitter):
            self._plot_segmented(ax, self.open_loop_iteration_times_raw, self.open_loop_iteration_times,
                    self.open_loop_jitter[:, 0],
                    label="x-cog (open loop)", linestyle="--", marker="s", linewidth=2, color="C0")
            self._plot_segmented(ax, self.open_loop_iteration_times_raw, self.open_loop_iteration_times,
                    self.open_loop_jitter[:, 1],
                    label="y-cog (open loop)", linestyle="--", marker="s", linewidth=2, color="C1")
            self._record_stat("jitter_x_open", self.open_loop_jitter[:, 0])
            self._record_stat("jitter_y_open", self.open_loop_jitter[:, 1])
        # Closed-loop jitter can be orders of magnitude below open-loop jitter,
        # which flattens it against zero on a linear axis -- switch to a log
        # y-axis when the plotted values span more than log_ratio.
        values = np.concatenate([np.ravel(j) for j in (self.jitter, self.open_loop_jitter) if _has_data(j)])
        values = values[np.isfinite(values) & (values > 0)]
        if values.size and values.max() / values.min() > log_ratio:
            ax.set_yscale("log")
        ax.set_ylabel("Jitter ($\\lambda/D$)")
        _format_time_axis(ax)
        ax.legend()
        fig_path = "PSFJitter.png"
        fig.savefig(fig_path, bbox_inches="tight")
        self._flag_figure("jitter")

    def MakeCoGStatsPlots(self):
        if not _has_data(self.short_exp_CoG_X):
            return

        # Split into closed- and open-loop runs (per-frame Is_Closed_Loop) and
        # use each regime's longest contiguous run -- concatenating separate,
        # non-adjacent runs before a Welch PSD would splice together samples
        # that were never actually adjacent in time, corrupting the estimate.
        if self.short_exp_is_closed_loop is not None:
            runs = find_status_runs(self.short_exp_is_closed_loop, self.short_exp_transition_buffer)
            closed_runs = [(s, e) for s, e, status in runs if status]
            open_runs = [(s, e) for s, e, status in runs if not status]
        else:
            closed_runs = [(0, len(self.short_exp_CoG_X))]
            open_runs = []

        fig_psd, ax_psd = plt.subplots()
        fig_cum, ax_cum = plt.subplots()
        f_for_xlim = None

        for runs, regime_label, linestyle in [(closed_runs, "closed loop", "-"), (open_runs, "open loop", "--")]:
            if not runs:
                continue
            start, end = max(runs, key=lambda run: run[1] - run[0])
            cog_x = self.short_exp_CoG_X[start:end]
            cog_y = self.short_exp_CoG_Y[start:end]

            f, x_psd = GetSignalPSD(cog_x, self.period)
            _, y_psd = GetSignalPSD(cog_y, self.period)
            f_for_xlim = f
            ax_psd.loglog(f, x_psd, label=f"x-cog ({regime_label})", color="C0", linestyle=linestyle)
            ax_psd.loglog(f, y_psd, label=f"y-cog ({regime_label})", color="C1", linestyle=linestyle)

            df = f[1] - f[0]
            cum_x = np.cumsum(x_psd) * df
            cum_y = np.cumsum(y_psd) * df
            cum_x = cum_x / cum_x[-1] * np.var(cog_x)
            cum_y = cum_y / cum_y[-1] * np.var(cog_y)
            ax_cum.loglog(f, np.sqrt(cum_x), label=f"x-cog ({regime_label})", color="C0", linestyle=linestyle)
            ax_cum.loglog(f, np.sqrt(cum_y), label=f"y-cog ({regime_label})", color="C1", linestyle=linestyle)

        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD ( $(\\lambda/D)^2/Hz$ )")
        ax_psd.legend()
        fig_psd.savefig("CoG_PSD.png", bbox_inches="tight")

        ax_cum.set_xlabel("Frequency (Hz)")
        ax_cum.set_ylabel("Cumulative jitter $(\\lambda/D)$")
        ax_cum.legend()
        if f_for_xlim is not None:
            ax_cum.set_xlim(f_for_xlim[1], f_for_xlim[-1])
        fig_cum.savefig("Cumulative_Jitter.png", bbox_inches="tight")
        self._flag_figure("cog_stats")
