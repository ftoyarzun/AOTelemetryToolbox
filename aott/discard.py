"""
Delete an observation file and its report, after asking for confirmation:

    python -m aott.discard [file] [--yes]

Without a file, takes the newest observation (analysed or not) in today's and
yesterday's UTC date folders of the [output] hdf5_dir. The report is the
report_dir/<date>/ao_report<file name>.pdf that AutomaticAnalysis.py writes.
"""
import argparse
from pathlib import Path

from aott.observation_files import newest_file, output_dirs


def report_files(hdf5_path, report_dir):
    """The ao_report<stem>.pdf files of an observation in the date folders of report_dir."""
    report_dir = Path(report_dir)
    if not report_dir.is_dir():
        return []
    return sorted(report_dir.glob(f"*/ao_report{Path(hdf5_path).stem}.pdf"))


def main():
    parser = argparse.ArgumentParser(description="Delete an observation file and its report.")
    parser.add_argument("file", nargs="?",
                        help="observation HDF5 file (default: the newest one, in today's and yesterday's "
                             "UTC date folders of the [output] hdf5_dir)")
    parser.add_argument("--yes", action="store_true", help="delete without asking")
    args = parser.parse_args()

    hdf5_dir, report_dir = output_dirs()
    hdf5_path = Path(args.file) if args.file else newest_file(hdf5_dir)
    if not hdf5_path.is_file():
        parser.error(f"{hdf5_path} does not exist")
    paths = [hdf5_path] + report_files(hdf5_path, report_dir)

    print("This will delete:")
    for path in paths:
        print(f"  {path}")
    if len(paths) == 1:
        print("(no report found)")
    if not args.yes and input("Are you sure? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing deleted.")
        return
    for path in paths:
        path.unlink()
        print(f"Deleted {path}")


if __name__ == "__main__":
    main()
