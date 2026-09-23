"""
Analyse one observation file and compile its report:

    python -m aott.AutomaticAnalysis [file.hdf5]

Without a file, the newest observation not yet analysed is used (see
aott.observation_files.newest_file). python -m aott.observe grabs a new
observation and calls analyze_and_report on it.
"""
from aott.PSF_Processing import PSF_Processing
from aott.Atmosphere_Characterization import Atmosphere_Characterization
from aott.AnalysisViewer import AnalysisViewer
from aott.frozen_flow_profiler import ProfilerInputError, profile_file, save_results
from aott.observation_files import newest_file, observation_span, output_dirs, telescope_name, utc_date
from aott.report import compile_report, copy_logo, new_run_dir
from pathlib import Path
import argparse
import h5py

from datetime import datetime, timezone


def analyze_and_report(file_name, report_dir):
    """
    Run the PSF analysis, the atmosphere characterization and the frozen-flow
    profiler on `file_name` (settings from config/analysis.toml), then compile
    its report to report_dir/<UTC date of the observation start>/. Returns the
    PDF path, or None if the compile failed.
    """
    file_name = Path(file_name)

    # UTC date of the observation start, like the hdf5_dir/<date> folder telemetry.py writes it to
    with h5py.File(file_name, "r") as file:
        span = observation_span(file)
        telescope = telescope_name(file) or "Unknown telescope"
    date = utc_date(span[0]) if span is not None else datetime.now(timezone.utc).strftime("%Y-%m-%d")

    psf = PSF_Processing(file_name)
    psf.SetPSFModel()
    psf.AnalyzeAllTheFile()

    atm_char = Atmosphere_Characterization(file_name)
    atm_char.AnalyzeAllTheFile()

    # Frozen-flow profiler, closed-loop runs only
    try:
        save_results(file_name, profile_file(file_name))
    except ProfilerInputError as e:
        print(f"Frozen-flow profiler skipped: {e}")

    # PNGs and report_data.json go to a temporary folder of this run's own,
    # where compile_report also compiles the template
    run_dir = new_run_dir("ao_report_")
    av = AnalysisViewer(file_name, figure_dir=run_dir)
    av.CreateAtmosphericAnalysisFigures()
    av.CreatePSFAnalysisFigures()
    av.SaveFigureManifest()

    inputs = {
        "AOtitle": file_name.stem,
        "telescope": telescope,
        "date": date,
        # Science.attrs["Target"], read by PSF_Processing
        "target": str(psf.target_name),
        "elevation": f"{psf.elevation:.1f}",
        "loop_gain": f"{atm_char.loop_gain:.3f}",
        "loop_leak": f"{atm_char.loop_leak:.3f}",
        "loop_freq": f"{atm_char.freq:.1f}",
        "logo": copy_logo(run_dir),
    }
    if av.VMag:
        inputs.update(VMag=f"{av.VMag:.2f}", RMag=f"{av.RMag:.2f}", JMag=f"{av.JMag:.2f}", HMag=f"{av.HMag:.2f}")
    else:
        inputs.update(VMag="none", RMag="none", JMag="none", HMag="none")

    return compile_report("ao_report.typ", run_dir,
                          Path(report_dir) / date / f"ao_report{file_name.stem}.pdf", inputs)


def main():
    parser = argparse.ArgumentParser(description="Analyse one observation and compile its report.")
    parser.add_argument("file", nargs="?",
                        help="observation HDF5 file (default: the newest one not yet analysed, in today's "
                             "and yesterday's UTC date folders of the [output] hdf5_dir)")
    args = parser.parse_args()

    # hdf5_dir and report_dir come from the [output] section of
    # config/data_grabber.toml, so this script and aott/telemetry.py agree on them.
    hdf5_dir, report_dir = output_dirs()
    file_name = Path(args.file) if args.file else newest_file(hdf5_dir, unanalysed_only=True)
    print(file_name)
    analyze_and_report(file_name, report_dir)


if __name__ == "__main__":
    main()
