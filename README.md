# AOTelemetryToolbox

AOTelemetryToolbox analyses the telemetry of an adaptive optics (AO) system together with the images of
its science camera. From one observation it estimates the atmospheric parameters (r0, seeing, tau0,
V0, the wind speed and strength of the turbulent layers) and characterises the PSF (Strehl ratio,
jitter). It then plots the results and compiles them into a PDF report.

The toolbox serves two purposes:

- **Monitoring while observing**: a short report right after each observation, and a report
  summarising the whole night.
- **Building up pre-analysed data**: every observation is stored as one self-describing HDF5 file,
  with the raw telemetry and the analysis results side by side, for later study.

The reports show plots and summary numbers only. They contain no thresholds, quality flags or written
interpretation: drawing conclusions is left to the AO scientist reading them.

## Installation

Requires Python 3.10 or later.

```
git clone https://github.com/ftoyarzun/AOTelemetryToolbox.git
cd AOTelemetryToolbox
pip install -e .
```

The Python dependencies (`h5py`, `numpy`, `scipy`, `matplotlib`, `astropy`, `astroquery`, `maoppy`) are
installed with the package. Three tools are not on PyPI and are installed separately:

- [**Typst**](https://typst.app), needed to compile the PDF reports. The `typst` command must be on
  your `PATH`. Without it, the analysis and the figures still run, and the report is skipped.
- [**dao**](https://github.com/Durham-Adaptive-Optics/daoBase), the shared-memory library of the
  real-time controller. Only acquisition needs it; analysing existing files does not.
- [**OOPAO**](https://github.com/cheritier/OOPAO), the AO simulator, only for the notebooks in
  `simulation/`.

## Quick start

### Analyse an observation

Everything downstream of acquisition works on an observation HDF5 file, so it runs on any machine and
with telemetry from any AO system, as long as the file follows the layout described in
[docs/hdf5_layout.md](docs/hdf5_layout.md).

1. Set the `[output]` folders in `config/data_grabber.toml`: `hdf5_dir`, where the observation files
   are, and `report_dir`, where the PDF reports go.
2. Run the analysis and the report:

   ```
   python -m aott.AutomaticAnalysis path/to/observation.hdf5
   ```

   Without a file, it takes the most recent observation in `hdf5_dir` that has not been analysed yet.
   The results are written into the HDF5 file itself, and the report to
   `report_dir/<date>/`.

To try it without real data, `simulation/DataGeneration.ipynb` simulates an observation with OOPAO
and writes it to `data/`.

### Nightly report

After the night, summarise every analysed observation of the last 20 hours in one report:

```
python -m aott.NightlyReport
```

### Acquire an observation

On a system whose real-time controller exposes its telemetry through dao shared memories, one command
grabs an observation, analyses it and compiles its report:

```
python -m aott.observe <target> <duration_s> [--no-simbad]
```

The target name is looked up in SIMBAD for its coordinates, elevation and magnitudes; `--no-simbad`
skips the lookup. `python -m aott.telemetry` runs the acquisition alone.

## Configuration

All configuration is in three TOML files in `config/`, which every script reads from that fixed
location:

| File                  | Contents |
|-----------------------|----------|
| `data_grabber.toml`   | This machine: the acquisition threads and the dao shared memories each one records, the ones read once (including calibration files), where each goes in the HDF5 file, which science camera the analysis uses, the output folders, and optionally the CPU cores to run on. |
| `instrument.toml`     | The instrument: observatory position, pupil diameter and central obstruction, DM geometry, WFS and science camera wavelengths and sampling. |
| `analysis.toml`       | Analysis settings: batch lengths, number of Zernike modes, radial orders used in the fits, PSD settings, and the frozen-flow profiler options. Ships with working defaults. |

The scripts refuse to start while a value they need is still `"TODO"`. `config/example_instrument.toml`
and the other files in `config/` are filled-in examples to copy from.

## What is computed

The telemetry is cut into batches (1 s by default). Each batch is entirely closed loop or entirely open
loop, as recorded by the real-time controller, and the two regimes are analysed differently.

**From the loop telemetry** (closed loop):

- **r0 and seeing**, from a von Kármán fit to the Zernike variances of the DM commands.
- **tau0, V0 and the turbulent layers**, from a multi-layer frozen-flow profiler: the speed, direction
  and relative strength of each layer, following Berdeu et al.
- **tau0 and V0 from the temporal autocorrelation** of each Zernike radial order, as a cross-check.
- **DM and WFS temporal PSDs** of the low-order modes, to spot vibrations.
- The outer scale L0, the effective loop gain and the loop delay are computed and stored, but not yet
  validated, so they are not shown in the reports.

In open loop, only the temporal PSDs of the WFS modes are computed.

**From the science camera**:

- **r0 and Strehl ratio**, from a fit of a PSF model to the long-exposure PSF of each batch
  (AO-corrected model in closed loop, turbulence-only model and no Strehl in open loop).
- **Tip-tilt jitter**, from the weighted centre of gravity of every frame, with its PSD and cumulative
  jitter.

r0, seeing and tau0 are given both along the line of sight and at zenith, at the reference wavelength
set in `instrument.toml`.

## Outputs

- **The observation HDF5 file**, with the raw data and the `Analysis` groups. Its layout is documented
  in [docs/hdf5_layout.md](docs/hdf5_layout.md).
- **The observation report** (`report_dir/<date>/`): atmospheric parameters, frozen-flow layers, jitter
  and centre-of-gravity PSD, Strehl ratio and example PSFs, DM/WFS PSD comparison. Sections without data
  (for example Strehl in an open-loop observation) are left out.
- **The nightly report**: one point per observation for every quantity over the night, and tables per
  target.

## Repository layout

```
aott/                   the package
  observe.py            acquisition + analysis + report in one command
  telemetry.py          acquisition from dao shared memories
  DataGrabber.py        reading the shared memories in parallel
  AutomaticAnalysis.py  analysis and report of one observation
  PSF_Processing.py     science camera analysis
  Atmosphere_Characterization.py, atmosphere_characterization_tools.py
                        loop telemetry analysis
  frozen_flow_profiler.py  multi-layer frozen-flow profiler (also runs on its own)
  AnalysisViewer.py     figures of one observation
  NightlyReport.py      nightly report
config/                 configuration files
docs/                   documentation
simulation/             notebooks: simulated observations, profiler prototyping
tests/                  tests on small synthetic observations
ao_report.typ, nightly_report.typ   Typst report templates
```

## Tests

```
pip install -e ".[dev]"
python -m pytest
```

The tests build small synthetic observations (one turbulent layer of known r0, speed and direction,
corrected by a simple integrator), run the whole analysis on them, check that the known parameters are
recovered, and compile both reports when Typst is available. They take about a minute and a half.

## References

- A. Berdeu et al., "Multi-layer frozen profiler from single conjugated adaptive optics telemetry",
  AO4ELT8. Basis of the frozen-flow profiler.
- J.-M. Conan, G. Rousset and P.-Y. Madec, "Wave-front temporal spectra in high-resolution imaging
  through turbulence", JOSA A 12, 1559 (1995). Frozen-flow model behind the autocorrelation estimator.
- R. J.-L. Fétick et al., "Physics-based model of the adaptive-optics-corrected point-spread
  function", A&A 628, A99 (2019), and the [maoppy](https://gitlab.lam.fr/lam-grd-public/maoppy)
  package. PSF model and fitting.
- C. T. Héritier et al., [OOPAO](https://github.com/cheritier/OOPAO), object-oriented Python AO
  simulator. Used for the simulated observations.

## License

MIT, see [LICENSE](LICENSE). Author: ftoyarzun.
