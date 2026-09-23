import h5py
import json

import numpy as np
import pylab as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta, timezone
from pathlib import Path

from aott.AnalysisViewer import _format_time_axis, utc_datetimes
from aott.observation_files import date_folders, hdf5_files, observation_span, output_dirs, telescope_name
from aott.report import compile_report, copy_logo, new_run_dir


# Every per-batch quantity the nightly report summarizes, as
# (key, dataset path, column or None, scale). The key is shared by the plots,
# the per-target tables and nightly_report.typ; scale converts to the units
# the per-observation report shows (PSF r0 is stored in m, shown in cm).
_QUANTITIES = [
    # r0 and tau0 at zenith, so observations at different elevations compare
    ("r0_wfs", "WFS/Analysis/r0_Zenith", None, 1.0),
    ("r0_psf_closed", "Science/Analysis/Long_Exposure/r0_Zenith", None, 100.0),
    ("r0_psf_open", "Science/Analysis/Long_Exposure_OpenLoop/r0_Zenith", None, 100.0),
    ("L0", "WFS/Analysis/L0", None, 1.0),
    ("tau0_frozen_flow", "WFS/Analysis/Frozen_Flow/tau0_Zenith", None, 1.0),
    ("tau0_autocorrelation", "WFS/Analysis/tau0_Autocorrelation_Zenith", None, 1.0),
    ("V0_frozen_flow", "WFS/Analysis/Frozen_Flow/V0", None, 1.0),
    ("V0_autocorrelation", "WFS/Analysis/V0_Autocorrelation", None, 1.0),
    ("sr", "Science/Analysis/Long_Exposure/sr_fit", None, 1.0),
    ("jitter_x_closed", "Science/Analysis/Short_Exposure/Jitter", 0, 1.0),
    ("jitter_y_closed", "Science/Analysis/Short_Exposure/Jitter", 1, 1.0),
    ("jitter_x_open", "Science/Analysis/Short_Exposure/Jitter_OpenLoop", 0, 1.0),
    ("jitter_y_open", "Science/Analysis/Short_Exposure/Jitter_OpenLoop", 1, 1.0),
]

# PNG filename per figures-manifest key, as nightly_report.typ loads it
_FIGURE_FILES = {
    "r0": "Nightly_r0.png",
    "L0": "Nightly_L0.png",
    "tau0": "Nightly_tau0.png",
    "V0": "Nightly_V0.png",
    "sr": "Nightly_Strehl.png",
    "jitter": "Nightly_Jitter.png",
}


def _robust_summary(values):
    """Median and interquartile range of the finite entries of `values`, or
    None if there are none."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return dict(median=float(median), q25=float(q25), q75=float(q75))


def _observation_time(file):
    """Unix timestamp at the middle of an observation (see observation_span), or None."""
    span = observation_span(file)
    return None if span is None else 0.5 * (span[0] + span[1])


class NightlyReport:
    """
    Summary of every analyzed observation from the last `window_hours`: one
    point per observation (the median over its batches, with the batch
    interquartile range as error bar) for the atmospheric parameters, Strehl
    ratio and jitter, plus per-target tables. Reads only the Analysis groups
    PSF_Processing and Atmosphere_Characterization already wrote -- files
    without them are listed in the report, not analyzed here.
    """

    def __init__(self, hdf5_dir, window_hours=20, end_time=None, telescope=None, figure_dir=None):
        self.hdf5_dir = Path(hdf5_dir)
        # Folder the PNGs and the manifest are written to, and the report compiled in
        self.figure_dir = Path(figure_dir) if figure_dir is not None else new_run_dir("nightly_report_")
        self.window_hours = window_hours
        # In UTC, like the date folders; a naive end_time is read as local time
        self.end_time = (end_time.astimezone(timezone.utc) if end_time is not None
                         else datetime.now(timezone.utc))
        self.start_time = self.end_time - timedelta(hours=window_hours)
        # Default: the Telescope and Instrument attrs of the observations
        self.telescope = telescope

        # One dict per analyzed observation in the window, sorted by time
        self.observations = []
        # (filename, reason) for files in the window that were left out
        self.skipped = []
        self.figures = {}

        self._collect_observations()

    def _candidate_files(self):
        """
        HDF5 files in the hdf5_dir/<UTC date> folders the window touches (an
        observing night spans two dates), or directly in hdf5_dir if none of
        those folders exist, for a flat folder of test data.
        """
        return hdf5_files(date_folders(self.hdf5_dir, self.start_time.timestamp(), self.end_time.timestamp()))

    def _collect_observations(self):
        start, end = self.start_time.timestamp(), self.end_time.timestamp()
        for path in self._candidate_files():
            try:
                with h5py.File(path, "r") as file:
                    obs_time = _observation_time(file)
                    if obs_time is None:
                        self.skipped.append((path.name, "no timestamps"))
                        continue
                    if not start <= obs_time <= end:
                        continue
                    if "WFS/Analysis" not in file and "Science/Analysis" not in file:
                        self.skipped.append((path.name, "no analysis results"))
                        continue
                    self.observations.append(self._load_observation(file, path, obs_time))
            except OSError as e:
                self.skipped.append((path.name, f"could not be opened ({e})"))
        self.observations.sort(key=lambda o: o["time"])
        if self.telescope is None:
            names = sorted({o["telescope"] for o in self.observations if o["telescope"]})
            self.telescope = ", ".join(names) if names else "Unknown telescope"

    def _load_observation(self, file, path, obs_time):
        sci_attrs = file["Science"].attrs if "Science" in file else {}
        mags = {}
        for band in "VRJH":
            mags[band] = float(sci_attrs[f"{band}mag"]) if f"{band}mag" in sci_attrs else None

        wavelength = None
        if "Science/Science_PSFs" in file and "Wavelength" in file["Science/Science_PSFs"].attrs:
            wavelength = float(file["Science/Science_PSFs"].attrs["Wavelength"])
        r0_reference_wvl = None
        if "Calibration" in file and "r0_reference_wvl" in file["Calibration"].attrs:
            r0_reference_wvl = float(file["Calibration"].attrs["r0_reference_wvl"])

        stats = {}
        for key, dataset, column, scale in _QUANTITIES:
            if dataset not in file or file[dataset].shape[0] == 0:
                continue
            values = file[dataset][:]
            if column is not None:
                values = values[:, column]
            summary = _robust_summary(values * scale)
            if summary is not None:
                stats[key] = summary

        return dict(
            file=path.name,
            telescope=telescope_name(file),
            target=str(sci_attrs["Target"]) if "Target" in sci_attrs else path.stem,
            time=obs_time,
            mags=mags,
            wavelength=wavelength,
            r0_reference_wvl=r0_reference_wvl,
            stats=stats,
        )

    def _plot_quantity(self, ax, key, label, color, marker):
        """One errorbar point per observation that has `key`: median, with
        the interquartile range over that observation's batches as error bar.
        Markers only -- nothing was measured between observations."""
        observations = [o for o in self.observations if key in o["stats"]]
        if not observations:
            return False
        times = utc_datetimes([o["time"] for o in observations])
        median = np.array([o["stats"][key]["median"] for o in observations])
        q25 = np.array([o["stats"][key]["q25"] for o in observations])
        q75 = np.array([o["stats"][key]["q75"] for o in observations])
        ax.errorbar(times, median, yerr=[median - q25, q75 - median], label=label,
                    color=color, marker=marker, linestyle="none", capsize=3)
        return True

    def _annotate_targets(self, ax):
        """Target names on a top axis, so color stays free to mean
        'estimator' as in the per-observation report: a minor tick at every
        observation, and a labeled tick at the first observation of each run
        of consecutive observations of the same target."""
        times = mdates.date2num(utc_datetimes([o["time"] for o in self.observations]))
        first_of_run = [i for i, o in enumerate(self.observations)
                        if i == 0 or o["target"] != self.observations[i - 1]["target"]]
        top = ax.secondary_xaxis("top")
        top.set_xticks([times[i] for i in first_of_run],
                       labels=[self.observations[i]["target"] for i in first_of_run])
        top.set_xticks(times, minor=True)
        top.tick_params(labelsize=7)
        for label in top.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("left")

    def _make_evolution_plot(self, figure_key, series, ylabel, log_ratio=None):
        """
        One nightly evolution plot from `series`, a list of
        (quantity key, legend label, color, marker). Skipped (no PNG, no
        manifest flag) if none of the series has data. With `log_ratio`, the
        y-axis switches to log when the plotted medians span more than that
        ratio, as in AnalysisViewer.MakeJitterPlot.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        plotted = [self._plot_quantity(ax, *s) for s in series]
        if not any(plotted):
            plt.close(fig)
            return

        if log_ratio is not None:
            values = np.array([o["stats"][s[0]]["median"] for s in series
                               for o in self.observations if s[0] in o["stats"]])
            values = values[np.isfinite(values) & (values > 0)]
            if values.size and values.max() / values.min() > log_ratio:
                ax.set_yscale("log")

        margin = timedelta(minutes=15)
        first, last = utc_datetimes([self.observations[0]["time"], self.observations[-1]["time"]])
        ax.set_xlim(first - margin, last + margin)
        ax.set_ylabel(ylabel)
        _format_time_axis(ax)
        self._annotate_targets(ax)
        ax.legend(fontsize=8)

        fig.savefig(self.figure_dir / _FIGURE_FILES[figure_key], bbox_inches="tight")
        plt.close(fig)
        self.figures[figure_key] = True

    def _common_wavelength(self, field, keys):
        """' @ N nm' for an axis label if every observation with one of the
        quantity `keys` has the same `field` wavelength (rounded to nm, so
        float noise in the stored attr doesn't split it), else ''."""
        wavelengths = {None if o[field] is None else round(o[field] * 1e9)
                       for o in self.observations if any(k in o["stats"] for k in keys)}
        if len(wavelengths) == 1 and None not in wavelengths:
            return f" @ {wavelengths.pop()} nm"
        return ""

    def MakeR0Plot(self):
        series = [
            ("r0_wfs", "r0 from AO telemetry", "C0", "o"),
            ("r0_psf_closed", "r0 from PSF processing (closed loop)", "C1", "o"),
            ("r0_psf_open", "r0 from PSF processing (open loop)", "C1", "s"),
        ]
        wavelength = self._common_wavelength("r0_reference_wvl", [s[0] for s in series])
        self._make_evolution_plot("r0", series, f"$r_0${wavelength}, zenith (cm)")

    def MakeL0Plot(self):
        self._make_evolution_plot("L0", [
            ("L0", "L0 from AO telemetry", "C0", "o"),
        ], "$L_0$ (m)")

    def MakeTau0Plot(self):
        # Frozen-flow profiler first, in C0: the main tau0/V0 estimator
        series = [
            ("tau0_frozen_flow", "tau0 from frozen-flow profiler", "C0", "o"),
            ("tau0_autocorrelation", "tau0 from autocorrelation (cross-check)", "C1", "o"),
        ]
        wavelength = self._common_wavelength("r0_reference_wvl", [s[0] for s in series])
        self._make_evolution_plot("tau0", series, f"$\\tau_0${wavelength}, zenith (ms)")

    def MakeV0Plot(self):
        self._make_evolution_plot("V0", [
            ("V0_frozen_flow", "V0 from frozen-flow profiler", "C0", "o"),
            ("V0_autocorrelation", "V0 from autocorrelation (cross-check)", "C1", "o"),
        ], "$V_0$ (m/s)")

    def MakeStrehlPlot(self):
        self._make_evolution_plot("sr", [
            ("sr", "Strehl ratio (closed loop)", "C0", "o"),
        ], "Strehl ratio" + self._common_wavelength("wavelength", ["sr"]) + " (%)")

    def MakeJitterPlot(self, log_ratio=10):
        # Color = axis, marker = loop status, as in AnalysisViewer.MakeJitterPlot
        self._make_evolution_plot("jitter", [
            ("jitter_x_closed", "x-cog (closed loop)", "C0", "o"),
            ("jitter_y_closed", "y-cog (closed loop)", "C1", "o"),
            ("jitter_x_open", "x-cog (open loop)", "C0", "s"),
            ("jitter_y_open", "y-cog (open loop)", "C1", "s"),
        ], "Jitter ($\\lambda/D$)", log_ratio=log_ratio)

    def CreateFigures(self):
        # MakeL0Plot is left out: the L0 estimator is not validated
        # (Validated=False attr)
        self.MakeR0Plot()
        self.MakeTau0Plot()
        self.MakeV0Plot()
        self.MakeStrehlPlot()
        self.MakeJitterPlot()

    def TargetSummaries(self):
        """
        One entry per target, in order of first observation: number of
        observations, magnitudes, and per quantity the median and
        interquartile range of the per-observation medians (each observation
        counts once, however many batches it had).
        """
        by_target = {}
        for o in self.observations:
            by_target.setdefault(o["target"], []).append(o)

        summaries = []
        for name, observations in by_target.items():
            mags = {band: next((o["mags"][band] for o in observations if o["mags"][band] is not None), None)
                    for band in "VRJH"}
            stats = {}
            for key, *_ in _QUANTITIES:
                medians = [o["stats"][key]["median"] for o in observations if key in o["stats"]]
                summary = _robust_summary(medians)
                if summary is not None:
                    stats[key] = summary
            summaries.append(dict(name=name, n_obs=len(observations), mags=mags, stats=stats))
        return summaries

    def SaveManifest(self, logo="none"):
        """Everything nightly_report.typ needs -- metadata, which figures
        exist, the per-target tables and the skipped files -- in one JSON,
        nightly_report_data.json next to the PNGs."""
        manifest = dict(
            telescope=self.telescope,
            date=self.end_time.strftime("%Y-%m-%d"),
            window_start=self.start_time.strftime("%Y-%m-%d %H:%M UTC"),
            window_end=self.end_time.strftime("%Y-%m-%d %H:%M UTC"),
            window_hours=self.window_hours,
            logo=logo,
            n_observations=len(self.observations),
            figures=self.figures,
            targets=self.TargetSummaries(),
            skipped=[dict(file=name, reason=reason) for name, reason in self.skipped],
        )
        with open(self.figure_dir / "nightly_report_data.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def CompileReport(self, report_dir):
        """
        Write the manifest, compile nightly_report.typ next to the PNGs and
        move the PDF to report_dir/<date>/ (see aott.report.compile_report).
        Returns the PDF path, or None if the compile failed.
        """
        self.SaveManifest(logo=copy_logo(self.figure_dir))
        date = self.end_time.strftime("%Y-%m-%d")
        return compile_report("nightly_report.typ", self.figure_dir,
                              Path(report_dir) / date / f"nightly_report_{date}.pdf")


if __name__ == "__main__":
    # Same [output] section of config/data_grabber.toml that AutomaticAnalysis.py reads
    _hdf5_dir, _report_dir = output_dirs()

    report = NightlyReport(_hdf5_dir, window_hours=20)
    print(f"{len(report.observations)} observations, {len(report.skipped)} skipped")
    report.CreateFigures()
    report.CompileReport(_report_dir)
