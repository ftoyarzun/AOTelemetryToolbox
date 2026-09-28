# Observation HDF5 file layout

Each observation is one HDF5 file. The acquisition (or any other source) writes the raw telemetry, the
analysis adds `Analysis` groups to the same file, and the viewer and reports read everything back from
it. This makes the file the interface between the stages: to analyse telemetry from an AO system that
AOTelemetryToolbox cannot grab from, write a file with the layout below and run
`python -m aott.AutomaticAnalysis <file>` on it.

`simulation/DataGeneration.ipynb` writes a file with this layout from an OOPAO simulation, and is the
reference implementation next to `aott/telemetry.py`.

Conventions used in the tables:

- `n_iter`: number of loop iterations recorded, `n_act`: number of DM actuators, `n_frames`: number of
  science frames, `n_batch`: number of analysis batches.
- Times are Unix timestamps in seconds (UTC). Wavelengths stored as attributes are in metres.
- Datasets in *italics* are optional. Everything else is required by the analysis, unless stated otherwise.
- The `Units` attribute, where present, gives the units of a dataset.

## Overview

```
/                         attrs: Instrument, Telescope
├── WFS/                  loop telemetry
│   └── Analysis/         atmosphere and loop results (written by the analysis)
│       ├── PSD_Comparison/
│       ├── Loop_Bandwidth/
│       ├── Open_Loop_PSD/
│       └── Frozen_Flow/
│           └── First_Batch/
├── Science/              science camera frames and target information
│   └── Analysis/         PSF results (written by the analysis)
│       ├── Long_Exposure/
│       ├── Long_Exposure_OpenLoop/
│       └── Short_Exposure/
└── Calibration/          calibration matrices and instrument geometry
```

## Raw data

These groups are written once, at acquisition.

### `/` (root)

| Attribute    | Type   | Meaning                                                  |
|--------------|--------|----------------------------------------------------------|
| `Instrument` | string | Name of the AO system, shown in the report title         |
| `Telescope`  | string | Name of the telescope, shown in the report title         |

### `WFS/`

| Attribute   | Type  | Meaning                                  |
|-------------|-------|------------------------------------------|
| `Loop_Gain` | float | Integrator gain of the loop              |
| `Loop_Leak` | float | Leak of the integrator                   |
| `Loop_Freq` | float | Loop frequency [Hz]                      |

| Dataset              | Shape                        | Meaning |
|----------------------|------------------------------|---------|
| `DM_commands`        | (n_iter, n_act)              | DM command at every loop iteration. Columns follow the row-major order of `DM_Map`. |
| `DM_TimeStamps`      | (n_iter,)                    | Time of each DM command [s]. |
| `WFS_measurements`   | (n_iter, n_act)              | WFS measurement at every loop iteration, in DM command space. |
| `loop_status`        | (n_iter,)                    | Loop command at every iteration. Nonzero means closed loop. This is the only source of the open/closed status. |
| `DM_Map`             | (n, n) or (n·n,)             | Actuator grid: ones where there is an actuator. Needed by the frozen-flow profiler. |
| `WFS_Images`         | (n_images, ny, nx)           | WFS camera frames. Attributes: `Gain`, `FPS` (frame rate [Hz], needed by the frozen-flow profiler), `Frame_Step` (one frame out of `Frame_Step` is kept). Not read by the analysis. |
| `WFS_TimeStamps`     | (n_images,)                  | Time of each kept WFS frame [s]. |
| `Valid_Pixel_Map`    | (ny, nx)                     | Valid pixels of the WFS camera. |
| *`Dark`*             | (ny, nx)                     | WFS camera dark. |
| *`Reference_Frame`*  | (ny, nx)                     | WFS reference frame. |
| *`DM_flat`*          | (n_act,)                     | DM flat command. |
| *`DM_offset`*        | (n_act,)                     | DM offset command. |

Older files may store the loop status as a `loop_status` attribute of `WFS/` instead of a dataset;
the analysis reads both. Older files may also call the WFS frames `OCAM_Images`.

### `Science/`

| Attribute   | Type   | Meaning |
|-------------|--------|---------|
| `Target`    | string | Target name, as typed at acquisition |
| `Elevation` | float  | Target elevation at the start of the acquisition [deg]. NaN if unknown; the zenith-corrected results are then NaN. |
| *`SIMBAD_ID`* | string | SIMBAD main identifier |
| *`RA`*, *`Dec`* | float | ICRS coordinates [deg] |
| *`Azimuth`* | float  | Azimuth at the start of the acquisition [deg] |
| *`Vmag`*, *`Rmag`*, *`Jmag`*, *`Hmag`* | float | Magnitudes from SIMBAD, NaN when SIMBAD has none |

| Dataset          | Shape                  | Meaning |
|------------------|------------------------|---------|
| `Science_PSFs`   | (n_frames, ny, nx)     | Science camera frames, background subtracted. Attributes: `Exposure_Time` [s], `FPS` [Hz], `Gain`, `Sampling` (pixels per λ/D at the calibration wavelength), `Wavelength` and `Bandpass` [m]. |
| `PSF_TimeStamps` | (n_frames,)            | Time of each science frame [s]. The science camera does not need to be synchronised with the loop. Older files store this as an attribute of `Science/`. |
| *`Dark`*         | (ny, nx)               | Science camera background, cropped like the frames. |

### `Calibration/`

| Attribute                          | Meaning |
|------------------------------------|---------|
| `Diameter`                         | Telescope pupil diameter [m] |
| `Obstruction_ratio`                | Central obstruction, as a ratio of diameters |
| `Science_Calibration_Wavelength`   | Wavelength at which the science `Sampling` was measured [m] |
| `AO_Calibration_Wavelength`        | Wavelength at which `Z2C` is expressed [m] |
| `SkyCalibPupilRatio`               | Ratio between the on-sky pupil and the calibration pupil on the DM |
| `Actuators_in_diameter`            | Number of actuators across the DM diameter |
| `Total_Number_Of_Actuators`        | Number of DM actuators |
| `Total_Number_Of_Controlled_Modes` | Number of columns of `M2C` |
| `r0_reference_wvl`                 | Wavelength that every r0, seeing and tau0 in the file refers to [m] |

| Dataset                  | Shape             | Meaning |
|--------------------------|-------------------|---------|
| `Z2C`                    | (n_act, n_zernike) | Zernike modes to DM commands, in radians at `AO_Calibration_Wavelength`. Columns are in Noll order starting at tip (j = 2). |
| `M2C`                    | (n_act, n_modes)  | Control modes to DM commands. |
| *`Interaction_Matrix`*   | any               | Interaction matrix. Attribute `Wavelength` [m]. |

## Analysis results

The analysis writes these groups. Re-running the analysis overwrites them, and deletes any optional
sub-group that the new run produces nothing for.

Every result is computed per batch of telemetry or science frames. Batches never straddle a change of
loop status, and the first frames after a change are skipped (the transition buffer), so each batch is
entirely closed loop or entirely open loop. `Iteration_Times` gives the start time of each batch [s].

All r0 values have a `*_Zenith` companion, scaled to zenith as r0 · sin(elevation)^(-3/5), and a `Seeing`
companion, 0.98 λ / r0 at `r0_reference_wvl` [arcsec]. tau0 scales like r0 to zenith.

### `WFS/Analysis/`

Computed from the DM commands, on closed-loop batches only. Batches overlap by half.

| Attribute                  | Meaning |
|----------------------------|---------|
| `Batch_Duration_s`         | Batch length [s] |
| `Transition_Buffer_Frames` | Loop iterations skipped after each change of loop status |
| `N_Zernike`                | Number of `Z2C` columns used |
| `Min_Radial_Order`, `Max_Radial_Order` | Radial orders used in the r0 fit |
| `Filter_Tip_Tilt`          | Whether tip and tilt were removed |
| `PSD_Nperseg`              | Segment length of the Welch PSDs |

| Dataset                        | Shape       | Units  | Meaning |
|--------------------------------|-------------|--------|---------|
| `Iteration_Times`              | (n_batch,)  | s      | Start of each batch |
| `r0`, `r0_Zenith`              | (n_batch,)  | cm     | Fried parameter from a von Kármán fit to the Zernike variances |
| `Seeing`, `Seeing_Zenith`      | (n_batch,)  | arcsec | Seeing from `r0` |
| `tau0_Autocorrelation`, `tau0_Autocorrelation_Zenith` | (n_batch,) | ms | Coherence time from the temporal autocorrelation of each radial order |
| `V0_Autocorrelation`           | (n_batch,)  | m/s    | Equivalent wind speed from the same method |
| `L0`                           | (n_batch,)  | m      | Outer scale from the von Kármán fit. `Validated = False`. |
| `Effective_Gain`               | (n_batch,)  |        | Loop gain from a fit of the AO transfer function to the PSDs. `Validated = False`. |
| `Measured_Loop_Delay`          | (n_batch,)  | frames | Loop delay from the same fit. `Validated = False`. |

Datasets with `Validated = False` have not been checked against simulations with known parameters and
are not shown in the reports.

#### `WFS/Analysis/PSD_Comparison/` (closed loop)

| Dataset           | Shape                        | Meaning |
|-------------------|------------------------------|---------|
| `Modes`           | (n_modes,)                   | Zernike mode indices compared (columns of `Z2C`) |
| `Frequency`       | (n_freq,)                    | Frequencies [Hz] |
| `DM_PSD`          | (n_batch, n_modes, n_freq)   | Temporal PSD of each mode, from the DM commands |
| `WFS_PSD`         | (n_batch, n_modes, n_freq)   | Temporal PSD of each mode, from the WFS measurements |
| `Iteration_Times` | (n_batch,)                   | Start of each batch [s] |

#### `WFS/Analysis/Loop_Bandwidth/` (closed loop, `Validated = False`)

| Dataset               | Shape                  | Meaning |
|-----------------------|------------------------|---------|
| `Radial_Orders`       | (n_orders,)            | Zernike radial orders |
| `Crossover_Frequency` | (n_batch, n_orders)    | Frequency where the DM and WFS PSDs cross [Hz] |
| `Iteration_Times`     | (n_batch,)             | Start of each batch [s] |

#### `WFS/Analysis/Open_Loop_PSD/` (open loop)

Open-loop batches only get this PSD. The WFS measurements in open loop are not reliable enough for r0,
tau0 or the loop parameters, so those are not computed.

| Dataset           | Shape                        | Meaning |
|-------------------|------------------------------|---------|
| `Modes`           | (n_modes,)                   | Zernike mode indices |
| `Frequency`       | (n_freq,)                    | Frequencies [Hz] |
| `PSD`             | (n_batch, n_modes, n_freq)   | Temporal PSD of each mode, from the WFS measurements |
| `Iteration_Times` | (n_batch,)                   | Start of each batch [s] |

#### `WFS/Analysis/Frozen_Flow/` (closed loop)

The multi-layer frozen-flow profiler (Berdeu et al.), the main tau0 and V0 estimator. It needs
`WFS/DM_Map` and the `FPS` attribute of `WFS/WFS_Images`; without them the group is not written.
`n_layers` is the maximum number of layers; unused layers are NaN.

| Dataset                     | Shape                              | Units  | Meaning |
|-----------------------------|------------------------------------|--------|---------|
| `Iteration_Times`           | (n_batch,)                         | s      | Start of each batch |
| `Batch_Frames`              | (n_batch,)                         | frames | Length of each batch |
| `N_Layers`                  | (n_batch,)                         |        | Number of layers found |
| `Velocity`                  | (n_batch, n_layers, 2)             | m/s    | Layer velocity along the two `DM_Map` axes |
| `Speed`                     | (n_batch, n_layers)                | m/s    | Layer speed |
| `Direction`                 | (n_batch, n_layers)                | deg    | atan2(v_axis1, v_axis0) |
| `Cn2_Fraction`              | (n_batch, n_layers)                |        | Relative strength of each layer |
| `V0`                        | (n_batch,)                         | m/s    | Equivalent wind speed |
| `r0`, `r0_Zenith`           | (n_batch,)                         | cm     | Fried parameter from the batch's DM commands |
| `tau0`, `tau0_Zenith`       | (n_batch,)                         | ms     | 0.314 r0 / V0 |
| `Seeing`, `Seeing_Zenith`   | (n_batch,)                         | arcsec | Seeing from `r0` |
| `Layer_Maps`                | (n_batch, n_layers, 2m, 2m)        |        | Fitted map of each layer |
| *`Correlation_Cube`*, *`Correlation_Weight`* | (n_batch, n_lags, 2m, 2m) | | Correlation cube of every batch, only when requested |

The profiler settings are stored as attributes of the group (`Actuator_Pitch_m`, `FPS`, `Signal`,
`Max_Lag_Frames`, `Batch_Size_Frames`, `Transition_Buffer_Frames`, `Low_Order_Modes_Removed`,
`Pupil_Radius_Pitch`, `Max_Speed`, `Elevation_deg`, ...).

`First_Batch/` keeps the full fit of the first batch, from which the report figures are drawn:
`Correlation_Cube`, `Model` and `Weight` (n_lags, 2m, 2m), `Velocity` (n_found, 2) [pixels/frame],
`Cn2` (n_found,), `Tracks` (n_found, n_lags, 2) [pixels], and the attributes `Start_Frame` and
`End_Frame`.

### `Science/Analysis/`

Computed from the science frames. Batches follow each other without overlap; the last batch of a run
may be shorter.

| Attribute                  | Meaning |
|----------------------------|---------|
| `Batch_Duration_s`         | Batch length [s] |
| `Transition_Buffer_Frames` | Science frames skipped after each change of loop status |
| `Frame_Rate_Hz`            | Science frame rate used, from `PSF_TimeStamps` |

#### `Science/Analysis/Long_Exposure/` (closed loop)

Each batch is averaged into a long-exposure PSF and fitted with an AO-corrected PSF model (`maoppy`
`Psfao`).

| Dataset                   | Shape                 | Units  | Meaning |
|---------------------------|-----------------------|--------|---------|
| `Iteration_Times`         | (n_batch,)            | s      | Start of each batch |
| `r0`, `r0_Zenith`         | (n_batch,)            | m      | Fried parameter from the PSF fit |
| `Seeing`, `Seeing_Zenith` | (n_batch,)            | arcsec | Seeing from `r0` |
| `sr_fit`                  | (n_batch,)            | %      | Strehl ratio from the fitted model. Attribute `Wavelength` [m]. |
| `sr_otf`                  | (n_batch,)            | %      | Strehl ratio from the OTF ratio against an annular pupil. Attributes `Wavelength` [m], `Obstruction_ratio`. |
| `psf_stack`               | (n_batch, npix, npix) |        | Normalised long-exposure PSF |
| `psf_model`               | (n_batch, npix, npix) |        | Fitted model PSF |

#### `Science/Analysis/Long_Exposure_OpenLoop/` (open loop)

The same as `Long_Exposure`, fitted with a turbulence-only model (`maoppy` `Turbulent`), and without
Strehl ratios: `Iteration_Times`, `r0`, `r0_Zenith`, `Seeing`, `Seeing_Zenith`, `psf_stack`, `psf_model`.

#### `Science/Analysis/Short_Exposure/`

| Dataset           | Shape         | Units | Meaning |
|-------------------|---------------|-------|---------|
| `CoG-X`, `CoG-Y`  | (n_frames,)   | λ/D   | Weighted centre of gravity of every frame, with the mean of each run removed. NaN for frames in a transition buffer. |
| `Is_Closed_Loop`  | (n_frames,)   |       | Loop status at each frame, from the nearest loop iteration |
| `Jitter`          | (n_batch, 2)  | λ/D   | Standard deviation of the CoG (x, y) within each closed-loop batch. Aligned with `Long_Exposure/Iteration_Times`. |
| `Jitter_OpenLoop` | (n_batch, 2)  | λ/D   | The same for open-loop batches. Aligned with `Long_Exposure_OpenLoop/Iteration_Times`. |

Attribute: `Transition_Buffer_Frames`.

## Reading a file

```python
import h5py
import numpy as np

with h5py.File("observation.hdf5", "r") as f:
    wvl = f["Calibration"].attrs["r0_reference_wvl"]
    times = f["WFS/Analysis/Iteration_Times"][:]
    r0_cm = f["WFS/Analysis/r0"][:]
    tau0_ms = f["WFS/Analysis/Frozen_Flow/tau0"][:]
    strehl = f["Science/Analysis/Long_Exposure/sr_fit"][:]

print(f"r0 at {wvl * 1e9:.0f} nm: median {np.nanmedian(r0_cm):.1f} cm")
```
