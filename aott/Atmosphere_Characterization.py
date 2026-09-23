import numpy as np
import h5py

from aott.config import AnalysisSettings
from aott.atmosphere_characterization_tools import (
    estimate_r0_L0,
    r0_at_zenith,
    tau0_at_zenith,
    seeing_arcsec,
    seeing_at_zenith,
    estimate_wind_gain_delay_from_psd,
    estimate_wind_speed_autocorrelation_cutoff,
    compute_zernike_psd,
    compute_zernike_psd_comparison,
    estimate_loop_bandwidth_from_psd_ratio,
    read_loop_status,
    find_status_runs,
)


def _filter_tip_tilt(commands, M2C):
    TT_modes = np.linalg.pinv(M2C[:, :2])
    TT_proj = commands @ TT_modes.T
    TT_commands = TT_proj @ M2C[:, :2].T
    return commands - TT_commands


class Atmosphere_Characterization:
    """
    Thin orchestrator around aott.atmosphere_characterization_tools: reads the
    WFS group of an observation HDF5 file, batches the DM-command/WFS-
    measurement telemetry per closed/open-loop run, calls the tools module's
    estimators, and writes the results to WFS/Analysis/*. Does not implement
    any atmosphere/AO-loop math itself -- see atmosphere_characterization_tools.py.

    Closed-loop batches get the full characterization, from the DM-derived
    Zernike modes (the loop's correction, a good proxy for the atmosphere it
    is correcting): r0, L0, tau0 and V0 (autocorrelation cutoff; the main
    tau0/V0 is the frozen-flow profiler's), gain/delay (PSD transfer-function fit), the DM-vs-WFS PSD
    comparison, and the DM/WFS PSD crossover frequency (Loop_Bandwidth, the
    commanded controller's crossover). L0, gain/delay and the crossover are
    written with a Validated=False attr and are not in the reports.

    Open-loop batches get none of that: with no active correction, a
    closed-loop-calibrated reconstructor applied to the WFS's raw signal is
    operating far outside the regime it was calibrated for (the WFS sees the
    full, uncorrected wavefront, not a small residual), so r0/L0/tau0/V0/gain/
    delay from open-loop WFS measurements are not trustworthy (on real
    telemetry, r0 came out ~15x too large). Open-loop batches therefore only
    get the one thing that doesn't require that reconstruction to be
    quantitatively accurate: the WFS-derived Zernike modes' own PSD,
    which AnalysisViewer plots alongside the closed-loop PSD comparison.
    """

    def __init__(self, file_name, batch_duration=None, filter_TT=None, psd_comparison_modes=None,
                 psd_nperseg=None, transition_buffer=None, n_zernike=None, min_radial_order=None,
                 max_radial_order=None):
        # Every setting not given comes from the [atmosphere] section of config/analysis.toml
        settings = AnalysisSettings(
            "atmosphere", batch_duration=batch_duration, filter_tip_tilt=filter_TT,
            psd_comparison_modes=psd_comparison_modes, psd_nperseg=psd_nperseg,
            transition_buffer=transition_buffer, n_zernike=n_zernike,
            min_radial_order=min_radial_order, max_radial_order=max_radial_order)
        self.file_name = file_name
        self.batch_duration = settings["batch_duration"]
        # Loop iterations skipped after each open/closed transition, so the
        # loop has time to settle into the new regime (see find_status_runs).
        self.transition_buffer = settings["transition_buffer"]
        self.filter_TT = settings["filter_tip_tilt"]
        self.psd_comparison_modes = np.asarray(settings["psd_comparison_modes"])
        self.psd_nperseg = settings["psd_nperseg"]
        self.n_zernike = settings["n_zernike"]
        self.min_radial_order = settings["min_radial_order"]
        self.max_radial_order = settings["max_radial_order"]

        with h5py.File(file_name, "r") as file:
            wfs_grp = file['WFS']
            self.dm_commands = wfs_grp['DM_commands'][:]
            self.dm_timestamps = wfs_grp['DM_TimeStamps'][:]
            self.wfs_measurements = wfs_grp['WFS_measurements'][:].squeeze()
            self.is_closed_loop_per_sample = read_loop_status(wfs_grp)
            self.loop_gain = wfs_grp.attrs['Loop_Gain']
            self.loop_leak = wfs_grp.attrs['Loop_Leak']
            self.freq = wfs_grp.attrs['Loop_Freq']
            self.period = 1 / self.freq

            calibration_grp = file['Calibration']
            self.wavelength = calibration_grp.attrs['AO_Calibration_Wavelength']
            self.M2C = calibration_grp['M2C'][:]
            self.Z2C = calibration_grp['Z2C'][:, :self.n_zernike]
            self.C2Z = np.linalg.pinv(self.Z2C)
            self.Diameter = calibration_grp.attrs['Diameter']
            self.r0_reference_wvl = calibration_grp.attrs["r0_reference_wvl"]
            # Elevation at the acquisition start [deg], NaN when unknown, for the zenith r0
            self.elevation = float(file['Science'].attrs.get('Elevation', np.nan)) if 'Science' in file else np.nan

        # batch_duration is in seconds
        self.batch_size = max(round(self.batch_duration * self.freq), 1)

        self.number_of_frames = self.dm_commands.shape[0]

    def _project_to_zernike(self, commands):
        if self.filter_TT:
            commands = _filter_tip_tilt(commands, self.M2C)
        commands = commands - commands.mean(axis=1, keepdims=True)
        return commands @ self.C2Z.T

    def _batch_timestamps(self, batch_start, batch_end):
        # DM_TimeStamps can be coarser than one batch (e.g. a wall-clock stamp
        # shared by many consecutive samples in simulated telemetry), which
        # would make the derived sample period zero -- build the per-sample
        # spacing from the authoritative Loop_Freq instead, anchored to this
        # batch's real start time so Iteration_Times still records true
        # wall-clock time.
        batch_start_time = self.dm_timestamps[batch_start]
        return batch_start_time + np.arange(batch_end - batch_start) * self.period

    def _process_closed_batch(self, batch_start, batch_end):
        dm_zernike = self._project_to_zernike(self.dm_commands[batch_start:batch_end])
        wfs_zernike = self._project_to_zernike(self.wfs_measurements[batch_start:batch_end])
        timestamps = self._batch_timestamps(batch_start, batch_end)
        # The same modes in radians at r0_reference_wvl, so that r0 and tau0
        # (0.31 r0/V0) are at that wavelength
        dm_zernike_ref = dm_zernike * (self.wavelength / self.r0_reference_wvl)

        r0l0 = estimate_r0_L0(dm_zernike_ref, self.Diameter, max_radial_order=self.max_radial_order,
                              min_radial_order=self.min_radial_order)
        autoc = estimate_wind_speed_autocorrelation_cutoff(dm_zernike_ref, timestamps, self.Diameter, r0=r0l0.r0)
        windgd = estimate_wind_gain_delay_from_psd(dm_zernike, timestamps, self.loop_leak)
        psd_comparison = compute_zernike_psd_comparison(
            dm_zernike, wfs_zernike, timestamps, self.psd_comparison_modes, nperseg=self.psd_nperseg)
        loop_bandwidth = estimate_loop_bandwidth_from_psd_ratio(
            dm_zernike, wfs_zernike, timestamps, nperseg=self.psd_nperseg)

        return dict(
            r0=r0l0.r0 * 100, L0=r0l0.L0,
            tau0_autocorrelation=(autoc.tau0 * 1000) if autoc.tau0 is not None else np.nan,
            V0_autocorrelation=autoc.V0,
            Effective_Gain=windgd.effective_gain,
            Measured_Loop_Delay=windgd.effective_delay,
            iteration_time=float(self.dm_timestamps[batch_start]),
            psd_comparison=psd_comparison, loop_bandwidth=loop_bandwidth,
        )

    def _process_open_batch(self, batch_start, batch_end):
        wfs_zernike = self._project_to_zernike(self.wfs_measurements[batch_start:batch_end])
        timestamps = self._batch_timestamps(batch_start, batch_end)

        psd = compute_zernike_psd(wfs_zernike, timestamps, self.psd_comparison_modes, nperseg=self.psd_nperseg)

        return dict(iteration_time=float(self.dm_timestamps[batch_start]), psd=psd)

    def AnalyzeAllTheFile(self):
        print('#####################')
        print('Analysing AO Telemetry')
        print('#####################')

        scalar_keys = [
            "r0", "L0", "Effective_Gain", "Measured_Loop_Delay",
            "tau0_autocorrelation", "V0_autocorrelation", "iteration_time",
        ]
        closed_results = {k: [] for k in scalar_keys}
        psd_comparisons = []
        loop_bandwidths = []
        open_psds = []
        open_iteration_times = []

        for run_start, run_end, is_closed in find_status_runs(self.is_closed_loop_per_sample,
                                                              self.transition_buffer):
            if run_end - run_start < self.batch_size:
                continue
            batch_start = run_start
            while batch_start + self.batch_size <= run_end:
                batch_end = batch_start + self.batch_size
                if is_closed:
                    entry = self._process_closed_batch(batch_start, batch_end)
                    for k in scalar_keys:
                        closed_results[k].append(entry[k])
                    psd_comparisons.append(entry["psd_comparison"])
                    loop_bandwidths.append(entry["loop_bandwidth"])
                else:
                    entry = self._process_open_batch(batch_start, batch_end)
                    open_psds.append(entry["psd"])
                    open_iteration_times.append(entry["iteration_time"])
                batch_start += self.batch_size // 2
                print(f'{batch_start} out of {self.number_of_frames} frames processed')

        for k in scalar_keys:
            closed_results[k] = np.array(closed_results[k])

        self.results = closed_results
        self.psd_comparisons = psd_comparisons
        self.loop_bandwidths = loop_bandwidths
        self.open_psds = open_psds
        self.open_iteration_times = np.array(open_iteration_times)

        self.SaveAnalysis()

    def SaveAnalysis(self):
        def write_or_replace(grp, name, data):
            data = np.asarray(data)
            if name in grp:
                dset = grp[name]
                if dset.shape == data.shape:
                    dset[:] = data
                else:
                    del grp[name]
                    dset = grp.create_dataset(name, data=data)
            else:
                dset = grp.create_dataset(name, data=data)
            return dset

        results = self.results
        with h5py.File(self.file_name, "a") as file:
            wfs_grp = file['WFS']
            analysis_grp = wfs_grp.require_group('Analysis')
            analysis_grp.attrs["Transition_Buffer_Frames"] = self.transition_buffer
            analysis_grp.attrs["Batch_Duration_s"] = self.batch_duration
            analysis_grp.attrs["N_Zernike"] = self.n_zernike
            analysis_grp.attrs["Min_Radial_Order"] = self.min_radial_order
            analysis_grp.attrs["Max_Radial_Order"] = self.max_radial_order
            analysis_grp.attrs["Filter_Tip_Tilt"] = self.filter_TT
            analysis_grp.attrs["PSD_Nperseg"] = self.psd_nperseg

            # Everything below (down to Loop_Bandwidth) is closed-loop batches
            # only -- see the class docstring for why open loop doesn't get
            # r0/L0/tau0/V0/gain/delay at all.
            write_or_replace(analysis_grp, "r0", results["r0"])
            analysis_grp["r0"].attrs["Units"] = "cm"
            write_or_replace(analysis_grp, "r0_Zenith", r0_at_zenith(results["r0"], self.elevation))
            analysis_grp["r0_Zenith"].attrs["Units"] = "cm"
            analysis_grp["r0_Zenith"].attrs["Elevation_deg"] = self.elevation
            seeing = seeing_arcsec(results["r0"] / 100, self.r0_reference_wvl)
            write_or_replace(analysis_grp, "Seeing", seeing)
            write_or_replace(analysis_grp, "Seeing_Zenith", seeing_at_zenith(seeing, self.elevation))
            for name in ("Seeing", "Seeing_Zenith"):
                analysis_grp[name].attrs["Units"] = "arcsec"
                analysis_grp[name].attrs["Wavelength"] = self.r0_reference_wvl
            analysis_grp["Seeing_Zenith"].attrs["Elevation_deg"] = self.elevation
            write_or_replace(analysis_grp, "L0", results["L0"])
            analysis_grp["L0"].attrs["Units"] = "m"
            # tau0/V0 of the structure-function estimator, in files analysed before it was removed
            for name in ("tau0", "V0"):
                if name in analysis_grp:
                    del analysis_grp[name]
            write_or_replace(analysis_grp, "Effective_Gain", results["Effective_Gain"])
            write_or_replace(analysis_grp, "Measured_Loop_Delay", results["Measured_Loop_Delay"])
            analysis_grp["Measured_Loop_Delay"].attrs["Units"] = "frames"
            write_or_replace(analysis_grp, "tau0_Autocorrelation", results["tau0_autocorrelation"])
            analysis_grp["tau0_Autocorrelation"].attrs["Units"] = "ms"
            write_or_replace(analysis_grp, "tau0_Autocorrelation_Zenith",
                             tau0_at_zenith(results["tau0_autocorrelation"], self.elevation))
            analysis_grp["tau0_Autocorrelation_Zenith"].attrs["Units"] = "ms"
            analysis_grp["tau0_Autocorrelation_Zenith"].attrs["Elevation_deg"] = self.elevation
            write_or_replace(analysis_grp, "V0_Autocorrelation", results["V0_autocorrelation"])
            analysis_grp["V0_Autocorrelation"].attrs["Units"] = "m/s"
            write_or_replace(analysis_grp, "Iteration_Times", results["iteration_time"])
            # Not validated against simulations with known parameters, so left
            # out of the reports
            for name in ("L0", "Effective_Gain", "Measured_Loop_Delay"):
                analysis_grp[name].attrs["Validated"] = False

            # An optional sub-group this run produced nothing for is deleted,
            # so a re-run with different batching never leaves a stale one.
            if self.psd_comparisons:
                psd_grp = analysis_grp.require_group("PSD_Comparison")
                write_or_replace(psd_grp, "Modes", self.psd_comparisons[0].modes)
                write_or_replace(psd_grp, "Frequency", self.psd_comparisons[0].frequency)
                write_or_replace(psd_grp, "DM_PSD", np.array([p.dm_psd for p in self.psd_comparisons]))
                write_or_replace(psd_grp, "WFS_PSD", np.array([p.wfs_psd for p in self.psd_comparisons]))
                write_or_replace(psd_grp, "Iteration_Times", results["iteration_time"])
            elif "PSD_Comparison" in analysis_grp:
                del analysis_grp["PSD_Comparison"]

            if self.loop_bandwidths:
                bw_grp = analysis_grp.require_group("Loop_Bandwidth")
                write_or_replace(bw_grp, "Radial_Orders", self.loop_bandwidths[0].radial_orders)
                write_or_replace(bw_grp, "Crossover_Frequency",
                                 np.array([b.crossover_frequency for b in self.loop_bandwidths]))
                write_or_replace(bw_grp, "Iteration_Times", results["iteration_time"])
                bw_grp.attrs["Description"] = (
                    "Frequency where the DM-derived and WFS-derived PSDs cross, per radial order: "
                    "the crossover of the commanded controller, set by the loop gain and leak")
                bw_grp.attrs["Validated"] = False
            elif "Loop_Bandwidth" in analysis_grp:
                del analysis_grp["Loop_Bandwidth"]

            if self.open_psds:
                ol_grp = analysis_grp.require_group("Open_Loop_PSD")
                write_or_replace(ol_grp, "Modes", self.open_psds[0].modes)
                write_or_replace(ol_grp, "Frequency", self.open_psds[0].frequency)
                write_or_replace(ol_grp, "PSD", np.array([p.psd for p in self.open_psds]))
                write_or_replace(ol_grp, "Iteration_Times", self.open_iteration_times)
            elif "Open_Loop_PSD" in analysis_grp:
                del analysis_grp["Open_Loop_PSD"]
