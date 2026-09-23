import h5py
import json
import os
import shutil
import subprocess

import numpy as np
import pylab as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from pathlib import Path

from aott.AnalysisViewer import _format_time_axis

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


# Every per-batch quantity the nightly report summarizes, as
# (key, dataset path, column or None, scale). The key is shared by the plots,
# the per-target tables and nightly_report.typ; scale converts to the units
# the per-observation report shows (PSF r0 is stored in m, shown in cm).
_QUANTITIES = [
    ("r0_wfs", "WFS/Analysis/r0", None, 1.0),
    ("r0_psf_closed", "Science/Analysis/Long_Exposure/r0", None, 100.0),
    ("r0_psf_open", "Science/Analysis/Long_Exposure_OpenLoop/r0", None, 100.0),
    ("L0", "WFS/Analysis/L0", None, 1.0),
    ("tau0", "WFS/Analysis/tau0", None, 1.0),
    ("tau0_autocorrelation", "WFS/Analysis/tau0_Autocorrelation", None, 1.0),
    ("V0", "WFS/Analysis/V0", None, 1.0),
    ("V0_autocorrelation", "WFS/Analysis/V0_Autocorrelation", None, 1.0),
    ("tau0_frozen_flow", "WFS/Analysis/Frozen_Flow/tau0", None, 1.0),
    ("V0_frozen_flow", "WFS/Analysis/Frozen_Flow/V0", None, 1.0),
    ("sr", "Science/Analysis/Long_Exposure/sr_fit", None, 1.0),
    ("jitter_x_closed", "Science/Analysis/Short_Exposure/Jitter", 0, 1.0),
    ("jitter_y_closed", "Science/Analysis/Short_Exposure/Jitter", 1, 1.0),
    ("jitter_x_open", "Science/Analysis/Short_Exposure/Jitter_OpenLoop", 0, 1.0),
    ("jitter_y_open", "Science/Analysis/Short_Exposure/Jitter_OpenLoop", 1, 1.0),
]

# PNG filename(s) per figures-manifest key, for RemoveFigureFiles -- an
# explicit whitelist, same reasoning as AnalysisViewer._FIGURE_FILES. The
# "Nightly_" prefix keeps them apart from the per-observation PNGs, which are
# written to the same working directory.
_FIGURE_FILES = {
    "r0": ["Nightly_r0.png"],
    "L0": ["Nightly_L0.png"],
    "tau0": ["Nightly_tau0.png"],
    "V0": ["Nightly_V0.png"],
    "sr": ["Nightly_Strehl.png"],
    "jitter": ["Nightly_Jitter.png"],
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
    """
    Unix timestamp at the middle of an observation: the midpoint of the first
    and last WFS/DM_TimeStamps sample (falling back to Science's
    PSF_TimeStamps dataset, or attr in older files). Only two samples are read,
    so checking a file against the report window stays cheap however long the
    observation is.
    """
    for path in ("WFS/DM_TimeStamps", "Science/PSF_TimeStamps"):
        if path in file:
            ts = file[path]
            if ts.shape[0] > 0:
                return 0.5 * (float(ts[0]) + float(ts[-1]))
    if "Science" in file and "PSF_TimeStamps" in file["Science"].attrs:
        ts = np.asarray(file["Science"].attrs["PSF_TimeStamps"], dtype=float)
        if ts.size > 0:
            return 0.5 * (float(ts[0]) + float(ts[-1]))
    return None


class NightlyReport:
    """
    Summary of every analyzed observation from the last `window_hours`: one
    point per observation (the median over its batches, with the batch
    interquartile range as error bar) for the atmospheric parameters, Strehl
    ratio and jitter, plus per-target tables. Reads only the Analysis groups
    PSF_Processing and Atmosphere_Characterization already wrote -- files
    without them are listed in the report, not analyzed here.
    """

    def __init__(self, hdf5_dir, window_hours=20, end_time=None, telescope="T152-Papyrus"):
        self.hdf5_dir = Path(hdf5_dir)
        self.window_hours = window_hours
        self.end_time = end_time if end_time is not None else datetime.now()
        self.start_time = self.end_time - timedelta(hours=window_hours)
        self.telescope = telescope

        # One dict per analyzed observation in the window, sorted by time
        self.observations = []
        # (filename, reason) for files in the window that were left out
        self.skipped = []
        self.figures = {}

        self._collect_observations()

    def _candidate_files(self):
        """
        HDF5 files in the hdf5_dir/<date> folders the window touches (an
        observing night spans two dates), or directly in hdf5_dir if none of
        those folders exist -- same fallback as AutomaticAnalysis.py, for a
        flat folder of test data.
        """
        folders = []
        day = self.start_time.date()
        while day <= self.end_time.date():
            folder = self.hdf5_dir / day.strftime("%Y-%m-%d")
            if folder.is_dir():
                folders.append(folder)
            day += timedelta(days=1)
        if not folders:
            folders = [self.hdf5_dir]
        return sorted(f for folder in folders for f in folder.iterdir()
                      if f.suffix.lower() in (".hdf5", ".h5"))

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

    def _load_observation(self, file, path, obs_time):
        sci_attrs = file["Science"].attrs if "Science" in file else {}
        mags = {}
        for band in "VRJH":
            mags[band] = float(sci_attrs[f"{band}mag"]) if f"{band}mag" in sci_attrs else None

        wavelength = None
        if "Science/Science_PSFs" in file and "Wavelength" in file["Science/Science_PSFs"].attrs:
            wavelength = float(file["Science/Science_PSFs"].attrs["Wavelength"])

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
            target=str(sci_attrs["Target"]) if "Target" in sci_attrs else path.stem,
            time=obs_time,
            mags=mags,
            wavelength=wavelength,
            stats=stats,
        )

    def _plot_quantity(self, ax, key, label, color, marker):
        """One errorbar point per observation that has `key`: median, with
        the interquartile range over that observation's batches as error bar.
        Markers only -- nothing was measured between observations."""
        observations = [o for o in self.observations if key in o["stats"]]
        if not observations:
            return False
        times = [datetime.fromtimestamp(o["time"]) for o in observations]
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
        times = [mdates.date2num(datetime.fromtimestamp(o["time"])) for o in self.observations]
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
        ax.set_xlim(datetime.fromtimestamp(self.observations[0]["time"]) - margin,
                    datetime.fromtimestamp(self.observations[-1]["time"]) + margin)
        ax.set_ylabel(ylabel)
        _format_time_axis(ax)
        self._annotate_targets(ax)
        ax.legend(fontsize=8)

        fig.savefig(_FIGURE_FILES[figure_key][0], bbox_inches="tight")
        plt.close(fig)
        self.figures[figure_key] = True

    def MakeR0Plot(self):
        self._make_evolution_plot("r0", [
            ("r0_wfs", "r0 from AO telemetry", "C0", "o"),
            ("r0_psf_closed", "r0 from PSF processing (closed loop)", "C1", "o"),
            ("r0_psf_open", "r0 from PSF processing (open loop)", "C1", "s"),
        ], "$r_0$ @ 500 nm (cm)")

    def MakeL0Plot(self):
        self._make_evolution_plot("L0", [
            ("L0", "L0 from AO telemetry", "C0", "o"),
        ], "$L_0$ (m)")

    def MakeTau0Plot(self):
        self._make_evolution_plot("tau0", [
            ("tau0", "tau0 from structure function", "C0", "o"),
            ("tau0_autocorrelation", "tau0 from autocorrelation", "C1", "o"),
            ("tau0_frozen_flow", "tau0 from frozen-flow profiler", "C2", "o"),
        ], "$\\tau_0$ @ 500 nm (ms)")

    def MakeV0Plot(self):
        self._make_evolution_plot("V0", [
            ("V0", "V0 from structure function", "C0", "o"),
            ("V0_autocorrelation", "V0 from autocorrelation", "C1", "o"),
            ("V0_frozen_flow", "V0 from frozen-flow profiler", "C2", "o"),
        ], "$V_0$ @ 500 nm (m/s)")

    def MakeStrehlPlot(self):
        # Science wavelength in the label only if every observation shares it
        # (rounded to nm, so float noise in the stored attr doesn't split it)
        wavelengths = {None if o["wavelength"] is None else round(o["wavelength"] * 1e9)
                       for o in self.observations if "sr" in o["stats"]}
        ylabel = "Strehl ratio"
        if len(wavelengths) == 1 and None not in wavelengths:
            ylabel += f" @ {wavelengths.pop()} nm"
        self._make_evolution_plot("sr", [
            ("sr", "Strehl ratio (closed loop)", "C0", "o"),
        ], ylabel)

    def MakeJitterPlot(self, log_ratio=10):
        # Color = axis, marker = loop status, as in AnalysisViewer.MakeJitterPlot
        self._make_evolution_plot("jitter", [
            ("jitter_x_closed", "x-cog (closed loop)", "C0", "o"),
            ("jitter_y_closed", "y-cog (closed loop)", "C1", "o"),
            ("jitter_x_open", "x-cog (open loop)", "C0", "s"),
            ("jitter_y_open", "y-cog (open loop)", "C1", "s"),
        ], "Jitter ($\\lambda/D$)", log_ratio=log_ratio)

    def CreateFigures(self):
        self.MakeR0Plot()
        self.MakeL0Plot()
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

    def SaveManifest(self, path="nightly_report_data.json", logo="none"):
        """Everything nightly_report.typ needs -- metadata, which figures
        exist, the per-target tables and the skipped files -- in one JSON."""
        manifest = dict(
            telescope=self.telescope,
            date=self.end_time.strftime("%Y-%m-%d"),
            window_start=self.start_time.strftime("%Y-%m-%d %H:%M"),
            window_end=self.end_time.strftime("%Y-%m-%d %H:%M"),
            window_hours=self.window_hours,
            logo=logo,
            n_observations=len(self.observations),
            figures=self.figures,
            targets=self.TargetSummaries(),
            skipped=[dict(file=name, reason=reason) for name, reason in self.skipped],
        )
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def RemoveFigureFiles(self):
        for key, produced in self.figures.items():
            if not produced:
                continue
            for fname in _FIGURE_FILES.get(key, []):
                if os.path.exists(fname):
                    os.remove(fname)

    def CompileReport(self, report_dir, template="nightly_report.typ"):
        """
        Write the manifest, compile `template` with Typst and move the PDF to
        report_dir/<date>/. Like AutomaticAnalysis.py, run from the repo root:
        the template loads the PNGs and the JSON by bare filename. Returns the
        PDF path, or None if the compile failed (PNGs are then left in place
        for debugging).
        """
        logo_path = Path("logo.png")
        self.SaveManifest(logo=logo_path.name if logo_path.exists() else "none")

        pdf_name = f"nightly_report_{self.end_time.strftime('%Y-%m-%d')}.pdf"
        cmd = ["typst", "compile", template, pdf_name]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("STDOUT:")
            print(e.stdout)
            print("\nSTDERR:")
            print(e.stderr)
            return None

        self.RemoveFigureFiles()
        save_folder = Path(report_dir) / self.end_time.strftime("%Y-%m-%d")
        save_folder.mkdir(parents=True, exist_ok=True)
        destination = save_folder / pdf_name
        shutil.move(pdf_name, str(destination))
        print(destination)
        return destination


if __name__ == "__main__":
    from aott.config import DATA_GRABBER_FILE

    # Same [output] section of config/data_grabber.toml that AutomaticAnalysis.py reads
    with open(DATA_GRABBER_FILE, "rb") as _f:
        _output_config = tomllib.load(_f)["output"]

    report = NightlyReport(_output_config["hdf5_dir"], window_hours=20)
    print(f"{len(report.observations)} observations, {len(report.skipped)} skipped")
    report.CreateFigures()
    report.CompileReport(_output_config["report_dir"])
