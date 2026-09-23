"""
Compiling a Typst report in a folder of its own. Each run writes its PNGs and
JSON to a fresh temporary folder, the template (and the logo, if the repo has
one) is copied in next to them, so image() and json() find them by bare
filename whatever the working directory is, and two runs never share files.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGO_FILE = REPO_ROOT / "logo.png"


def new_run_dir(prefix):
    """A fresh temporary folder for one report's PNGs and JSON."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def copy_logo(run_dir):
    """Copy the repo's logo.png into `run_dir`; returns the name the template
    should load, or "none" when there is no logo."""
    if not LOGO_FILE.exists():
        return "none"
    shutil.copy(LOGO_FILE, Path(run_dir) / LOGO_FILE.name)
    return LOGO_FILE.name


def compile_report(template, run_dir, destination, inputs=None):
    """
    Compile the repo's `template` (a .typ file name) inside `run_dir`, with
    `inputs` passed as --input key=value, and move the PDF to `destination`,
    creating its folder. On success `run_dir` is deleted and the destination
    returned; if typst is missing or the compile fails, the error is printed,
    `run_dir` is kept for debugging and None is returned.
    """
    run_dir = Path(run_dir)
    typst = shutil.which("typst")
    if typst is None:
        print(f"typst not found on the PATH; figures kept in {run_dir}")
        return None

    shutil.copy(REPO_ROOT / template, run_dir / template)
    pdf = run_dir / (Path(template).stem + ".pdf")
    cmd = [typst, "compile", str(run_dir / template), str(pdf)]
    for key, value in (inputs or {}).items():
        cmd += ["--input", f"{key}={value}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"typst compile failed; figures kept in {run_dir}")
        return None

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pdf), str(destination))
    shutil.rmtree(run_dir)
    print(destination)
    return destination
