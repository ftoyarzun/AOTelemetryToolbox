"""
Grab one observation, analyse it and compile its report, in one call:

    python -m aott.observe <target> <duration_s> [--no-simbad]

The same as python -m aott.telemetry followed by python -m aott.AutomaticAnalysis
on the file it wrote. Returns when the PDF is written. Observatory machine only
(the grab needs dao).
"""
import argparse

from aott.AutomaticAnalysis import analyze_and_report
from aott.observation_files import output_dirs
from aott.telemetry import acquire


def main():
    parser = argparse.ArgumentParser(description="Grab one observation, analyse it and compile its report.")
    parser.add_argument("target", type=str, help="name of the target, as SIMBAD knows it")
    parser.add_argument("duration", type=float, help="acquisition time in seconds")
    parser.add_argument("--no-simbad", action="store_true",
                        help="grab without the SIMBAD query: no magnitudes or coordinates, NaN elevation")
    args = parser.parse_args()

    # Checked before the grab, so an incomplete [output] section doesn't waste one
    _, report_dir = output_dirs()
    hdf5_path = acquire(args.target, args.duration, args.no_simbad)
    try:
        analyze_and_report(hdf5_path, report_dir)
    except BaseException:
        print(f"The analysis failed; the observation is saved in {hdf5_path}. "
              f"Rerun it with: python -m aott.AutomaticAnalysis {hdf5_path}")
        raise


if __name__ == "__main__":
    main()
