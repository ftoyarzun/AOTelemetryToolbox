"""
Multi-layer frozen-flow profiler for closed-loop SCAO telemetry.

Method: A. Berdeu et al., "Multi-layer frozen profiler from single conjugated
adaptive optics telemetry: novel approach and potential applications",
AO4ELT8 (2025), with the details the paper leaves open (correlation weights,
peak tracking, extended layer maps, normalised v0) taken from A. Berdeu's
MATLAB implementation (get_2Dt_corr_cube.m, fit_max_peak_v.m,
fit_multi_layer.m).

For each closed-loop batch of an observation HDF5 file:

1. the pseudo-open loop (or the DM commands alone) is optionally
   low-passed onto the first Zernike modes of Z2C, cleaned of the first
   Zernike modes (default tip, tilt, defocus) and of each actuator's mean,
   placed on the actuator grid given by WFS/DM_Map, and turned into
   Fried-geometry x/y gradients, optionally only inside a pupil annulus;
2. the pupil-weighted 2D+t autocorrelation of the gradients is computed
   (paper eqs. 2-4);
3. the cube is modelled as a sum of 2D maps each translating at its own
   velocity, fitted one layer at a time (paper Algorithm 1): the brightest
   peak of the residual gives the velocity, a weighted least-squares solve
   gives the layer's map, and the value of the map at zero shift is the
   layer's relative strength;
4. v0 = (sum_l (Cn2_l / sum Cn2) |v_l|^(5/3))^(3/5) and tau0 = 0.314 r0 / v0.

Results go to WFS/Analysis/Frozen_Flow. Velocities are in the DM_Map array
axes (axis 0, axis 1), not rotated to the sky.

    python -m aott.frozen_flow_profiler [file.hdf5] [--signal pol|dm] [--plot [last|all]] ...

Without a file, the observation with the latest start time in today's and
yesterday's UTC date folders of the [output] hdf5_dir of
config/data_grabber.toml is used.
"""
import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pylab as plt
import matplotlib.dates as mdates
import scipy.fft as sfft
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, lsqr, MatrixRankWarning

from aott.config import AnalysisSettings
from aott.observation_files import newest_file, output_dirs
from aott.atmosphere_characterization_tools import (
    estimate_r0_L0,
    find_status_runs,
    r0_at_zenith,
    read_loop_status,
    reconstruct_pseudo_open_loop,
    seeing_arcsec,
    seeing_at_zenith,
    tau0_at_zenith,
)

# Voxels of the correlation cube whose fraction of overlapping pupil pairs is
# below this are left out of the fit
VALID_OVERLAP = 0.01
# Peaks are tracked up to this fraction of the radius of the region valid at every lag
TRACK_RADIUS_FRACTION = 0.95
# Half-size of the peak-fitting box at lag 0, before it is resized from the fitted width
INITIAL_BOX = 3

_FIGURE_PREFIX = "FrozenFlow_"


class ProfilerInputError(Exception):
    """The file lacks something the profiler needs (DM_Map, the WFS FPS, a consistent actuator count)."""


# ---------------------------------------------------------------------------
# Slope cube
# ---------------------------------------------------------------------------

def dm_map_to_grid(dm_map, n_actuators):
    """
    Boolean (n, n) actuator map. DM_Map may be stored as the (n, n) map or
    flattened (n*n,), row-major either way.
    """
    dm_map = np.asarray(dm_map).astype(bool)
    if dm_map.ndim == 1:
        n = int(round(np.sqrt(dm_map.size)))
        if n * n != dm_map.size:
            raise ProfilerInputError(f"WFS/DM_Map has {dm_map.size} elements, not a square grid")
        dm_map = dm_map.reshape(n, n)
    if dm_map.sum() != n_actuators:
        raise ProfilerInputError(f"WFS/DM_Map has {dm_map.sum()} actuators, DM_commands has {n_actuators}")
    return dm_map


def remove_low_order(commands, low_order_z2c):
    """Remove the projection of each sample on the given Zernike command vectors, then each actuator's mean."""
    if low_order_z2c.shape[1] > 0:
        coefficients = commands @ np.linalg.pinv(low_order_z2c).T
        commands = commands - coefficients @ low_order_z2c.T
    return commands - commands.mean(axis=0)


def zernike_low_pass(commands, z2c):
    """Keep only the projection of each sample on the given Zernike command vectors."""
    return commands @ np.linalg.pinv(z2c).T @ z2c.T


def pupil_mask(dm_map, pupil_radius, obstruction_ratio):
    """
    Actuators of `dm_map` inside the annulus obstruction_ratio * pupil_radius
    <= r <= pupil_radius, r measured in actuator pitches from the grid centre.
    """
    c = np.arange(dm_map.shape[0]) - (dm_map.shape[0] - 1) / 2
    r = np.hypot(c[:, None], c[None, :])
    return dm_map & (r <= pupil_radius) & (r >= obstruction_ratio * pupil_radius)


def commands_to_grid(commands, dm_map):
    """(n_samples, n_act) commands to (n_samples, n, n) maps, zero outside the actuators."""
    phi = np.zeros((commands.shape[0],) + dm_map.shape)
    phi[:, dm_map] = commands
    return phi


def fried_gradients(phi, mask):
    """
    Fried-geometry x/y gradients of (n_samples, n, n) maps: the difference
    along one axis summed over the two neighbours along the other. Returns the
    (2, n_samples, n-1, n-1) slope cube, zero outside the gradient mask, and
    that mask (all four corners valid).
    """
    d0 = phi[:, 1:, :] - phi[:, :-1, :]
    d0 = d0[:, :, 1:] + d0[:, :, :-1]
    d1 = phi[:, :, 1:] - phi[:, :, :-1]
    d1 = d1[:, 1:, :] + d1[:, :-1, :]
    grad_mask = mask[1:, 1:] & mask[:-1, 1:] & mask[1:, :-1] & mask[:-1, :-1]
    slopes = np.stack([d0, d1]) * grad_mask
    return slopes, grad_mask


# ---------------------------------------------------------------------------
# Weighted correlation cube (paper eqs. 2-4)
# ---------------------------------------------------------------------------

def weighted_correlation_cube(slopes, mask, max_lag):
    """
    alpha(d, t) = sum_z [p s_z * p s_z](d, t) / sum_z [p * p s_z^2](d, t)
    w(d, t)     = [p * p](d, t) / sum p

    `*` is the correlation over (space, time), `p` the gradient mask repeated
    over time. `slopes` is (n_z, n_samples, m, m) and already zero outside
    `mask`. Returns alpha and w for lags 0..max_lag, shaped
    (max_lag + 1, 2m, 2m), with zero spatial shift at index m. alpha is the
    least-squares scale between the cube and its shifted copy, and w the
    fraction of valid voxel pairs behind each value.
    """
    n_samples = slopes.shape[1]
    m0, m1 = mask.shape
    shape = (n_samples + max_lag, 2 * m0, 2 * m1)

    def correlate(a, b):
        fa = sfft.rfftn(a, s=shape, workers=-1)
        fb = sfft.rfftn(b, s=shape, workers=-1)
        c = sfft.irfftn(np.conj(fa) * fb, s=shape, workers=-1)[:max_lag + 1]
        return np.fft.fftshift(c, axes=(1, 2))

    p = np.broadcast_to(mask.astype(float), (n_samples, m0, m1))
    numerator = sum(correlate(s, s) for s in slopes)
    denominator = sum(correlate(p, s ** 2) for s in slopes)
    weight = correlate(p, p) / p.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(denominator > 0, numerator / denominator, 0.0)
    return alpha, weight


# ---------------------------------------------------------------------------
# Multi-layer fit (paper Algorithm 1)
# ---------------------------------------------------------------------------

def _shift_operator(velocity, n_lag, size, center, valid):
    """
    Bilinear interpolation operator I(v) from a layer map to the valid voxels
    of a (n_lag, size, size) cube: voxel (t, c0, c1) reads the map at
    (c0 - v0 t, c1 - v1 t). The map grid extends far enough that no voxel
    reads outside it. Returns the sparse operator, the map coordinate of the
    map's [0, 0] pixel and the map shape.
    """
    coords = np.arange(size) - center
    lags = np.arange(n_lag)
    q0 = np.broadcast_to(coords[None, :, None] - velocity[0] * lags[:, None, None], (n_lag, size, size))
    q1 = np.broadcast_to(coords[None, None, :] - velocity[1] * lags[:, None, None], (n_lag, size, size))
    origin = (int(np.floor(q0.min())), int(np.floor(q1.min())))
    map_shape = (int(np.floor(q0.max())) - origin[0] + 2, int(np.floor(q1.max())) - origin[1] + 2)

    q0 = q0.ravel()[valid] - origin[0]
    q1 = q1.ravel()[valid] - origin[1]
    i0 = np.floor(q0).astype(int)
    i1 = np.floor(q1).astype(int)
    f0 = q0 - i0
    f1 = q1 - i1

    rows = np.tile(np.arange(len(q0)), 4)
    cols = np.concatenate([i0 * map_shape[1] + i1, i0 * map_shape[1] + i1 + 1,
                           (i0 + 1) * map_shape[1] + i1, (i0 + 1) * map_shape[1] + i1 + 1])
    vals = np.concatenate([(1 - f0) * (1 - f1), (1 - f0) * f1, f0 * (1 - f1), f0 * f1])
    operator = csr_matrix((vals, (rows, cols)), shape=(len(q0), map_shape[0] * map_shape[1]))
    return operator, origin, map_shape


def _solve_layer_map(operator, residual, weight):
    """
    Weighted least-squares layer map (paper eq. 11):
    argmin_a || residual - I a ||_W = (I^T W I)^-1 I^T W residual,
    solved on the map pixels that at least one voxel reads.
    """
    itw = operator.T.multiply(weight[None, :]).tocsr()
    itwi = (itw @ operator).tocsc()
    itwc = itw @ residual
    seen = np.asarray(itwi.sum(axis=1)).ravel() > 0

    layer_map = np.zeros(operator.shape[1])
    if not seen.any():
        return layer_map
    lhs = itwi[seen][:, seen]
    rhs = itwc[seen]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            solution = spsolve(lhs, rhs)
        except MatrixRankWarning:
            solution = np.full(rhs.shape, np.nan)
    if not np.all(np.isfinite(solution)):
        solution = lsqr(lhs, rhs)[0]
    layer_map[seen] = solution
    return layer_map


def track_peak_velocity(cube, weight, radius, center, max_step=None):
    """
    Velocity of the brightest correlation peak, in pixels per frame.

    At lag 0 a 2D Gaussian centred at zero shift (amplitude, width, offset) is
    fitted in a box of half-size INITIAL_BOX; the box is then resized to
    floor(3 width) + 1. At each following lag the amplitude, centre and offset
    are refitted in the box around the previous centre, with the width fixed.
    The free amplitude absorbs the layer's decorrelation, the free offset the
    background of the other layers. The zero-shift pixel is left out of every
    fit (measurement noise only correlates at zero shift and zero lag).
    Tracking stops when the centre leaves `radius` minus the box diagonal, or
    at the last lag. The velocity is the least-squares slope, through the
    origin, of the centre against the lag. `max_step` (pixels per frame), if
    given, bounds how far the centre moves from one lag to the next, which
    keeps the track from jumping to another peak.

    Returns (velocity (2,), track (n_tracked, 2)).
    """
    n_lag, size = cube.shape[0], cube.shape[1]
    coords = np.arange(size) - center
    box = INITIAL_BOX
    amplitude, sigma, offset = 1.0, box / 2, 0.0
    x0 = x1 = 0.0
    box_limit = np.floor(radius - np.sqrt(2) * box)
    track = []

    lag = 0
    while x0 ** 2 + x1 ** 2 < box_limit ** 2 and lag < n_lag:
        w = weight[lag].copy()
        w[center, center] = 0
        i0 = np.argmin(np.abs(coords - x0))
        i1 = np.argmin(np.abs(coords - x1))
        rows = np.unique(np.clip(i0 + np.arange(-box, box + 1), 0, size - 1))
        cols = np.unique(np.clip(i1 + np.arange(-box, box + 1), 0, size - 1))
        c0 = coords[rows][:, None]
        c1 = coords[cols][None, :]
        data = cube[lag][np.ix_(rows, cols)]
        sqrt_w = np.sqrt(np.clip(w[np.ix_(rows, cols)], 0, None))

        if lag == 0:
            def residuals(p):
                return (sqrt_w * (p[0] * np.exp(-0.5 * (c0 ** 2 + c1 ** 2) / p[1] ** 2) + p[2] - data)).ravel()
            lower, upper = [0, 1e-3, -1], [1, np.inf, 1]
            start = np.clip([amplitude, sigma, offset], lower, upper)
            amplitude, sigma, offset = least_squares(
                residuals, start, bounds=(lower, upper), ftol=1e-2, xtol=1e-2, max_nfev=100).x
            box = int(np.floor(3 * sigma)) + 1
            box_limit = np.floor(radius - np.sqrt(2) * box)
        else:
            def residuals(p):
                return (sqrt_w * (p[0] * np.exp(-0.5 * ((c0 - p[1]) ** 2 + (c1 - p[2]) ** 2) / sigma ** 2)
                                  + p[3] - data)).ravel()
            step = box if max_step is None else min(box, max_step)
            lower = [0, x0 - step, x1 - step, -1]
            upper = [1, x0 + step, x1 + step, 1]
            start = np.clip([amplitude, x0, x1, offset], lower, upper)
            amplitude, x0, x1, offset = least_squares(
                residuals, start, bounds=(lower, upper), ftol=1e-2, xtol=1e-2, max_nfev=100).x
        track.append((x0, x1))
        lag += 1

    track = np.array(track).reshape(-1, 2)
    lags = np.arange(len(track))
    if len(track) < 2:
        return np.zeros(2), track
    velocity = lags @ track / (lags @ lags)
    return np.where(np.isfinite(velocity), velocity, 0.0), track


@dataclass
class FrozenFlowResult:
    velocity: np.ndarray    # (n_layers, 2) pixels per frame, DM_Map axes
    cn2: np.ndarray         # (n_layers,) map value at zero shift, sorted descending
    maps: np.ndarray        # (n_layers, size, size) layer maps over the cube footprint
    tracks: list            # per layer, the (n_tracked, 2) peak centres
    cube: np.ndarray        # (n_lag, size, size) correlation cube, zero outside the fit mask
    model: np.ndarray       # (n_lag, size, size) sum of the layer models
    weight: np.ndarray      # (n_lag, size, size) fit weight


def fit_frozen_layers(alpha, weight, n_layers=6, min_peak=0.0, max_step=None):
    """
    Greedy multi-layer fit of a correlation cube (paper Algorithm 1). Layers
    are added one at a time; each addition fits the new layer first, then
    refits every earlier one against the residual of all the others. After
    each addition the layers are sorted by strength, and layers keep being
    added while the weakest one is above `min_peak` and there are fewer than
    `n_layers`. `max_step` is passed to track_peak_velocity.
    """
    n_lag, size = alpha.shape[0], alpha.shape[1]
    center = size // 2
    fit_mask = weight > VALID_OVERLAP
    cube = np.where(fit_mask, alpha, 0.0)
    w = np.where(fit_mask, weight, 0.0)
    valid = fit_mask.ravel()
    cube_valid = cube.ravel()[valid]
    w_valid = w.ravel()[valid]

    always_valid = fit_mask.all(axis=0)
    radius = TRACK_RADIUS_FRACTION * np.sqrt(always_valid.sum() / np.pi)

    layers = []
    while True:
        layers.append(dict(velocity=np.zeros(2), cn2=0.0, model=np.zeros(valid.sum()),
                           map=None, origin=(0, 0), track=np.zeros((0, 2))))
        for i in [len(layers) - 1] + list(range(len(layers) - 1)):
            others = sum((layer["model"] for k, layer in enumerate(layers) if k != i),
                         np.zeros(valid.sum()))
            residual_valid = cube_valid - others
            residual = np.zeros(cube.size)
            residual[valid] = residual_valid
            residual = residual.reshape(cube.shape)

            velocity, track = track_peak_velocity(residual, w, radius, center, max_step)
            operator, origin, map_shape = _shift_operator(velocity, n_lag, size, center, valid)
            layer_map = _solve_layer_map(operator, residual_valid, w_valid).reshape(map_shape)

            layers[i] = dict(velocity=velocity, cn2=layer_map[-origin[0], -origin[1]],
                             model=operator @ layer_map.ravel(), map=layer_map, origin=origin, track=track)

        layers.sort(key=lambda layer: layer["cn2"], reverse=True)
        if not (layers[-1]["cn2"] > min_peak and len(layers) < n_layers):
            break

    model = np.zeros(cube.size)
    model[valid] = sum(layer["model"] for layer in layers)
    maps = []
    for layer in layers:
        start0 = -center - layer["origin"][0]
        start1 = -center - layer["origin"][1]
        maps.append(layer["map"][start0:start0 + size, start1:start1 + size])

    return FrozenFlowResult(
        velocity=np.array([layer["velocity"] for layer in layers]),
        cn2=np.array([layer["cn2"] for layer in layers]),
        maps=np.array(maps),
        tracks=[layer["track"] for layer in layers],
        cube=cube, model=model.reshape(cube.shape), weight=w,
    )


def equivalent_wind_speed(cn2, speed):
    """v0 = (sum_l (Cn2_l / sum Cn2) |v_l|^(5/3))^(3/5)."""
    total = np.sum(cn2)
    if total <= 0:
        return np.nan
    return np.sum(cn2 / total * np.abs(speed) ** (5 / 3)) ** (3 / 5)


# ---------------------------------------------------------------------------
# File-level driver
# ---------------------------------------------------------------------------

def lag_for_speed(min_speed, lag_pitches, pitch, fps):
    """Frames a layer moving at `min_speed` (m/s) takes to cross `lag_pitches` actuator pitches."""
    return int(np.ceil(lag_pitches * pitch * fps / min_speed))


def default_pupil_radius(actuators_in_diameter):
    """Radius of the outermost actuator centres, (n - 1) / 2 pitches: the pupil edge in Fried geometry."""
    return (actuators_in_diameter - 1) / 2


def closed_loop_batches(is_closed, batch_size, min_length, transition_buffer):
    """
    (start, end) frame ranges to profile: in each closed-loop run, batches of
    min(run length, batch_size) frames stepping by half a batch, never across
    a run boundary. Runs shorter than `min_length` frames, and open-loop runs,
    are skipped.
    """
    batches = []
    for run_start, run_end, closed in find_status_runs(is_closed, transition_buffer):
        if not closed or run_end - run_start < min_length:
            continue
        length = min(run_end - run_start, batch_size)
        start = run_start
        while start + length <= run_end:
            batches.append((start, start + length))
            start += max(length // 2, 1)
    return batches


def _none_if_string(value, word):
    """None where `value` is the string `word` (any case), else `value` unchanged."""
    return None if isinstance(value, str) and value.lower() == word else value


def profile_file(file_name, signal=None, frame_delay=None, dm_sign=None, batch_size=None, max_lag=None,
                 min_speed=None, lag_pitches=None, min_run_lags=None, n_layers=None, min_peak=None,
                 transition_buffer=None, low_order_modes=None, n_zernike=None, pupil_radius=None, max_speed=None):
    """
    Run the profiler on every closed-loop batch of an observation file.

    Every setting not given (None) comes from the [frozen_flow] section of
    config/analysis.toml, where max_lag = "auto", pupil_radius = "none" and
    max_speed = "none" stand for the None values described below. The r0 of
    each batch uses the Zernike count and radial orders of its [atmosphere]
    section, the same as Atmosphere_Characterization.

    `signal` is "dm" (DM commands alone) or "pol" (pseudo-open loop:
    wfs(t + frame_delay) + dm_sign * dm(t); dm_sign = -1 matches an integrator
    dm(t+1) = dm(t) - g wfs(t+1)).

    Each closed-loop run is cut into batches of min(run length, batch_size)
    frames stepping by half a batch (closed_loop_batches); open-loop runs are
    skipped. `max_lag` (frames) defaults to the time a layer at `min_speed`
    (m/s) takes to move `lag_pitches` actuator pitches, and runs shorter than
    `min_run_lags` * max_lag frames are skipped. `low_order_modes` is the number of
    leading Z2C columns (Zernike modes from tip on) removed from every sample.
    Removing them keeps tip/tilt vibrations out of the correlation, but the
    per-frame subtraction also adds a pupil-fixed, non-moving pattern to the
    slopes, which the fit sees as a static layer.

    `n_zernike` > 0 keeps only the projection of the signal on the first
    n_zernike columns of Z2C, which drops actuator-scale structure that does
    not follow the turbulence. `pupil_radius` (actuator pitches from the
    DM_Map centre) keeps only the slopes inside the pupil annulus (inner
    radius from Calibration.attrs["Obstruction_ratio"]), leaving out the
    actuators the WFS doesn't see; "auto" takes
    (Actuators_in_diameter - 1) / 2, None keeps every actuator. `max_speed`
    (m/s), if given, bounds the peak's move from one lag to the next.

    Returns a dict with the per-batch results (SI units except r0 in cm and
    tau0 in ms) and the list of FrozenFlowResult.
    """
    settings = AnalysisSettings(
        "frozen_flow", signal=signal, frame_delay=frame_delay, dm_sign=dm_sign, batch_size=batch_size,
        max_lag=max_lag, min_speed=min_speed, lag_pitches=lag_pitches, min_run_lags=min_run_lags,
        n_layers=n_layers, min_peak=min_peak, transition_buffer=transition_buffer,
        low_order_modes=low_order_modes, n_zernike=n_zernike, pupil_radius=pupil_radius, max_speed=max_speed)
    signal, frame_delay, dm_sign = settings["signal"], settings["frame_delay"], settings["dm_sign"]
    batch_size, min_speed, lag_pitches = settings["batch_size"], settings["min_speed"], settings["lag_pitches"]
    min_run_lags, n_layers, min_peak = settings["min_run_lags"], settings["n_layers"], settings["min_peak"]
    transition_buffer, low_order_modes = settings["transition_buffer"], settings["low_order_modes"]
    n_zernike = settings["n_zernike"]
    max_lag = _none_if_string(settings["max_lag"], "auto")
    pupil_radius = _none_if_string(settings["pupil_radius"], "none")
    max_speed = _none_if_string(settings["max_speed"], "none")
    r0_settings = AnalysisSettings("atmosphere")

    with h5py.File(file_name, "r") as file:
        wfs_grp = file["WFS"]
        if "DM_Map" not in wfs_grp:
            raise ProfilerInputError(f"{file_name}: no WFS/DM_Map, can't place the actuators on a grid")
        if "WFS_Images" not in wfs_grp or "FPS" not in wfs_grp["WFS_Images"].attrs:
            raise ProfilerInputError(f"{file_name}: no WFS/WFS_Images FPS attribute")
        try:
            is_closed = read_loop_status(wfs_grp)
        except KeyError as e:
            raise ProfilerInputError(f"{file_name}: {e.args[0]}")
        dm_commands = wfs_grp["DM_commands"][:]
        wfs_measurements = wfs_grp["WFS_measurements"][:].squeeze()
        dm_timestamps = wfs_grp["DM_TimeStamps"][:]
        dm_map = dm_map_to_grid(wfs_grp["DM_Map"][:], dm_commands.shape[1])
        fps = float(wfs_grp["WFS_Images"].attrs["FPS"])

        calibration_grp = file["Calibration"]
        z2c_all = calibration_grp["Z2C"][:]
        obstruction_ratio = float(calibration_grp.attrs.get("Obstruction_ratio", 0.0))
        diameter = float(calibration_grp.attrs["Diameter"])
        actuators_in_diameter = calibration_grp.attrs.get("Actuators_in_diameter")
        ao_wavelength = calibration_grp.attrs["AO_Calibration_Wavelength"]
        r0_reference_wvl = calibration_grp.attrs["r0_reference_wvl"]
        # Elevation at the acquisition start [deg], NaN when unknown, for the zenith r0
        elevation = float(file["Science"].attrs.get("Elevation", np.nan)) if "Science" in file else np.nan

    n = dm_map.shape[0]
    if actuators_in_diameter is not None and actuators_in_diameter != n:
        print(f"WARNING: DM_Map is {n} actuators across, Calibration says {actuators_in_diameter}")
    pitch = diameter / n
    pixel_per_frame_to_mps = pitch * fps
    if max_lag is None:
        max_lag = lag_for_speed(min_speed, lag_pitches, pitch, fps)
    if pupil_radius == "auto":
        pupil_radius = default_pupil_radius(actuators_in_diameter if actuators_in_diameter is not None else n)
    z2c = z2c_all[:, :r0_settings["n_zernike"]]
    low_order_z2c = z2c_all[:, :low_order_modes]
    low_pass_z2c = z2c_all[:, :n_zernike]
    if n_zernike > z2c_all.shape[1]:
        print(f"WARNING: Z2C has {z2c_all.shape[1]} modes, fewer than n_zernike = {n_zernike}")
    c2z = np.linalg.pinv(z2c)
    slope_map = dm_map if pupil_radius is None else pupil_mask(dm_map, pupil_radius, obstruction_ratio)
    max_step = None if max_speed is None else max_speed / pixel_per_frame_to_mps

    batches =closed_loop_batches(is_closed, batch_size, min_run_lags * max_lag, transition_buffer)

    keys = ["Iteration_Times", "Batch_Frames", "Velocity", "Speed", "Direction", "Cn2_Fraction", "N_Layers",
            "V0", "r0", "r0_Zenith", "tau0", "tau0_Zenith", "Seeing", "Seeing_Zenith"]
    results = {k: [] for k in keys}
    fits = []
    for k, (start, end) in enumerate(batches):
        print(f"Frozen flow: batch {k + 1}/{len(batches)} (frames {start}-{end})")
        dm = dm_commands[start:end]
        if signal == "pol":
            commands, _ = reconstruct_pseudo_open_loop(dm_sign * dm, wfs_measurements[start:end],
                                                       np.arange(end - start), frame_delay)
        else:
            commands = dm
        if n_zernike > 0:
            commands = zernike_low_pass(commands, low_pass_z2c)
        commands = remove_low_order(commands, low_order_z2c)
        slopes, grad_mask = fried_gradients(commands_to_grid(commands, dm_map), slope_map)
        alpha, weight = weighted_correlation_cube(slopes, grad_mask, max_lag)
        fit = fit_frozen_layers(alpha, weight, n_layers, min_peak, max_step)
        fits.append(fit)

        velocity = np.full((n_layers, 2), np.nan)
        cn2 = np.full(n_layers, np.nan)
        velocity[:len(fit.cn2)] = fit.velocity * pixel_per_frame_to_mps
        cn2[:len(fit.cn2)] = fit.cn2
        speed = np.hypot(velocity[:, 0], velocity[:, 1])
        v0 = equivalent_wind_speed(fit.cn2, speed[:len(fit.cn2)])

        dm_zernike = (dm - dm.mean(axis=1, keepdims=True)) @ c2z.T
        try:
            r0 = estimate_r0_L0(dm_zernike, diameter, max_radial_order=r0_settings["max_radial_order"],
                                min_radial_order=r0_settings["min_radial_order"]).r0
            r0 *= (r0_reference_wvl / ao_wavelength) ** (6 / 5)
        except (RuntimeError, ValueError):
            r0 = np.nan

        results["Iteration_Times"].append(float(dm_timestamps[start]))
        results["Batch_Frames"].append(end - start)
        results["Velocity"].append(velocity)
        results["Speed"].append(speed)
        results["Direction"].append(np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0])))
        results["Cn2_Fraction"].append(cn2)
        results["N_Layers"].append(len(fit.cn2))
        results["V0"].append(v0)
        results["r0"].append(r0 * 100)
        results["r0_Zenith"].append(float(r0_at_zenith(r0 * 100, elevation)))
        results["tau0"].append(0.314 * r0 / v0 * 1000 if v0 > 0 else np.nan)
        results["tau0_Zenith"].append(float(tau0_at_zenith(results["tau0"][-1], elevation)))
        results["Seeing"].append(float(seeing_arcsec(r0, r0_reference_wvl)))
        results["Seeing_Zenith"].append(float(seeing_at_zenith(results["Seeing"][-1], elevation)))

    for k in keys:
        results[k] = np.array(results[k])
    settings = dict(Actuator_Pitch_m=pitch, FPS=fps, Signal=signal, Frame_Delay=frame_delay, DM_Sign=dm_sign,
                    Max_Lag_Frames=max_lag, Min_Speed=min_speed, Lag_Pitches=lag_pitches,
                    Batch_Size_Frames=batch_size, Min_Run_Lags=min_run_lags, Transition_Buffer_Frames=transition_buffer,
                    Low_Order_Modes_Removed=low_order_modes, Zernike_Low_Pass_Modes=n_zernike,
                    Pupil_Radius_Pitch=np.nan if pupil_radius is None else pupil_radius,
                    Max_Speed=np.nan if max_speed is None else max_speed,
                    N_Zernike_r0=r0_settings["n_zernike"], Min_Radial_Order_r0=r0_settings["min_radial_order"],
                    Max_Radial_Order_r0=r0_settings["max_radial_order"], Elevation_deg=elevation,
                    Direction_Frame="DM_Map axes (axis 0, axis 1)")
    return dict(results=results, fits=fits, batches=batches, settings=settings)


def _write_or_replace(grp, name, data):
    data = np.asarray(data)
    if name in grp:
        dset = grp[name]
        if dset.shape == data.shape:
            dset[:] = data
            return dset
        del grp[name]
    return grp.create_dataset(name, data=data)


def save_results(file_name, profile, save_cube=False):
    """Write WFS/Analysis/Frozen_Flow, or delete it when there is no batch to write."""
    results = profile["results"]
    fits = profile["fits"]
    units = dict(Iteration_Times="s", Batch_Frames="frames", Velocity="m/s", Speed="m/s", Direction="deg",
                 V0="m/s", r0="cm", r0_Zenith="cm", tau0="ms", tau0_Zenith="ms", Seeing="arcsec",
                 Seeing_Zenith="arcsec")
    with h5py.File(file_name, "a") as file:
        analysis_grp = file["WFS"].require_group("Analysis")
        if not fits:
            if "Frozen_Flow" in analysis_grp:
                del analysis_grp["Frozen_Flow"]
            return
        grp = analysis_grp.require_group("Frozen_Flow")
        for key, values in results.items():
            dset = _write_or_replace(grp, key, values)
            if key in units:
                dset.attrs["Units"] = units[key]

        n_layers = results["Cn2_Fraction"].shape[1]
        size = fits[0].maps.shape[-1]
        maps = np.full((len(fits), n_layers, size, size), np.nan)
        for k, fit in enumerate(fits):
            maps[k, :len(fit.maps)] = fit.maps
        _write_or_replace(grp, "Layer_Maps", maps)

        # The first batch in full, so its figures can be redrawn from the file
        first = fits[0]
        first_grp = grp.require_group("First_Batch")
        _write_or_replace(first_grp, "Correlation_Cube", first.cube.astype(np.float32))
        _write_or_replace(first_grp, "Model", first.model.astype(np.float32))
        _write_or_replace(first_grp, "Weight", first.weight.astype(np.float32))
        _write_or_replace(first_grp, "Velocity", first.velocity)
        first_grp["Velocity"].attrs["Units"] = "pixels/frame"
        _write_or_replace(first_grp, "Cn2", first.cn2)
        longest = max(len(t) for t in first.tracks)
        tracks = np.full((len(first.tracks), longest, 2), np.nan)
        for k, track in enumerate(first.tracks):
            tracks[k, :len(track)] = track
        _write_or_replace(first_grp, "Tracks", tracks)
        first_grp["Tracks"].attrs["Units"] = "pixels"
        first_grp.attrs["Start_Frame"], first_grp.attrs["End_Frame"] = profile["batches"][0]

        if save_cube:
            _write_or_replace(grp, "Correlation_Cube", np.array([fit.cube for fit in fits]))
            _write_or_replace(grp, "Correlation_Weight", np.array([fit.weight for fit in fits]))
        else:
            for name in ("Correlation_Cube", "Correlation_Weight"):
                if name in grp:
                    del grp[name]
        for key, value in profile["settings"].items():
            grp.attrs[key] = value


def read_first_batch(frozen_flow_grp):
    """
    The FrozenFlowResult of the first batch, plus its speeds (m/s) and
    directions (deg), from a WFS/Analysis/Frozen_Flow group written by
    save_results.
    """
    first_grp = frozen_flow_grp["First_Batch"]
    cn2 = first_grp["Cn2"][:]
    n = len(cn2)
    tracks = [t[np.isfinite(t[:, 0])] for t in first_grp["Tracks"][:]]
    fit = FrozenFlowResult(
        velocity=first_grp["Velocity"][:], cn2=cn2, maps=frozen_flow_grp["Layer_Maps"][0, :n],
        tracks=tracks, cube=first_grp["Correlation_Cube"][:].astype(float),
        model=first_grp["Model"][:].astype(float), weight=first_grp["Weight"][:].astype(float),
    )
    return fit, frozen_flow_grp["Speed"][0, :n], frozen_flow_grp["Direction"][0, :n]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_correlation(fit, fig_path, n_lags_shown=3):
    """Data, model and residual of the correlation cube at zero lag and a few later lags, with the fitted peak positions."""
    n_lag, size = fit.cube.shape[0], fit.cube.shape[1]
    center = size // 2
    fastest = np.max(np.hypot(fit.velocity[:, 0], fit.velocity[:, 1]))
    last = n_lag - 1 if fastest == 0 else min(n_lag - 1, int(round(center / fastest)))
    lags = [0] + [max(1, int(round(last * k / n_lags_shown))) for k in range(1, n_lags_shown + 1)]
    vmax = fit.cn2.max() if fit.cn2.max() > 0 else 1.0
    extent = [-center - 0.5, size - center - 0.5, size - center - 0.5, -center - 0.5]

    fig, axes = plt.subplots(3, len(lags), figsize=(3 * len(lags), 9), squeeze=False)
    for col, lag in enumerate(lags):
        for row, (label, cube) in enumerate([("Data", fit.cube), ("Model", fit.model),
                                             ("Residual", fit.cube - fit.model)]):
            ax = axes[row, col]
            ax.imshow(cube[lag], vmin=-0.5 * vmax, vmax=vmax, extent=extent, cmap="viridis")
            for k, v in enumerate(fit.velocity):
                ax.plot(v[1] * lag, v[0] * lag, "x", color=f"C{k}", ms=8, mew=2)
            ax.set_title(f"{label}, lag {lag}")
            if col == 0:
                ax.set_ylabel("Shift, axis 0 [pitch]")
            if row == 2:
                ax.set_xlabel("Shift, axis 1 [pitch]")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_layer_maps(fit, speed, fig_path):
    """One panel per fitted layer map."""
    n = len(fit.maps)
    size = fit.maps.shape[-1]
    center = size // 2
    vmax = fit.cn2.max() if fit.cn2.max() > 0 else 1.0
    extent = [-center - 0.5, size - center - 0.5, size - center - 0.5, -center - 0.5]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4), squeeze=False)
    for k in range(n):
        ax = axes[0, k]
        ax.imshow(fit.maps[k], vmin=-0.5 * vmax, vmax=vmax, extent=extent, cmap="viridis")
        ax.set_title(f"Layer {k + 1}: {speed[k]:.1f} m/s\n$C_n^2$ = {fit.cn2[k]:.3f}", color=f"C{k}")
        ax.set_xlabel("axis 1 [pitch]")
        if k == 0:
            ax.set_ylabel("axis 0 [pitch]")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_layer_profile(cn2, speed, direction, fig_path):
    """C_n^2 and speed per layer, and speed against direction."""
    n = np.count_nonzero(np.isfinite(cn2))
    layers = np.arange(1, n + 1)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(layers, cn2[:n], "o", color="C0")
    ax.set_xlabel("Layer")
    ax.set_ylabel("$C_n^2$", color="C0")
    ax.set_xticks(layers)
    ax2 = ax.twinx()
    ax2.plot(layers, speed[:n], "s", color="C3")
    ax2.set_ylabel("Speed [m/s]", color="C3")

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    for k in range(n):
        ax.plot(np.radians(direction[k]), speed[k], "o", color=f"C{k}",
                ms=4 + 20 * max(cn2[k], 0) / max(np.nanmax(cn2), 1e-12))
    ax.set_title("Speed [m/s] vs direction (DM_Map axes)", fontsize=10, pad=20)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def plot_evolution(results, fig_path):
    """Speed of every layer (marker size ~ C_n^2) and V0, against time."""
    times = [datetime.fromtimestamp(t, timezone.utc) for t in results["Iteration_Times"]]
    speed = results["Speed"]
    cn2 = np.nan_to_num(np.clip(results["Cn2_Fraction"], 0, None))
    scale = 80 / max(cn2.max(), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(speed.shape[1]):
        ax.scatter(times, speed[:, k], s=5 + scale * cn2[:, k], color=f"C{k}", alpha=0.7, label=f"Layer {k + 1}")
    ax.plot(times, results["V0"], "k.-", label="$V_0$")
    ax.set_ylabel("Speed [m/s]")
    locator = mdates.AutoDateLocator(tz=timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    ax.set_xlabel("Time (UTC)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def make_plots(profile, which="last", prefix=_FIGURE_PREFIX):
    """Save the PNGs for the last batch (or every batch) and the time evolution over the file."""
    results = profile["results"]
    fits = profile["fits"]
    if not fits:
        return
    indices = range(len(fits)) if which == "all" else [len(fits) - 1]
    for k in indices:
        suffix = f"_{k:03d}" if which == "all" else ""
        fit = fits[k]
        n = len(fit.cn2)
        plot_correlation(fit, f"{prefix}correlation{suffix}.png")
        plot_layer_maps(fit, results["Speed"][k][:n], f"{prefix}layer_maps{suffix}.png")
        plot_layer_profile(results["Cn2_Fraction"][k], results["Speed"][k], results["Direction"][k],
                           f"{prefix}layers{suffix}.png")
    plot_evolution(results, f"{prefix}evolution.png")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer frozen-flow profiler (Berdeu et al.). Every option left out takes its "
                    "value from the [frozen_flow] section of config/analysis.toml.")
    parser.add_argument("file", nargs="?",
                        help="observation HDF5 file (default: the newest one in today's and yesterday's "
                             "UTC date folders of the [output] hdf5_dir)")
    parser.add_argument("--signal", choices=["pol", "dm"], help="DM commands alone or pseudo-open loop")
    parser.add_argument("--frame-delay", type=int, help="loop delay of the pseudo-open loop [frames]")
    parser.add_argument("--dm-sign", type=float, help="sign of the DM commands in the pseudo-open loop")
    parser.add_argument("--batch-size", type=int,
                        help="maximum frames per batch; shorter closed-loop runs are one batch")
    parser.add_argument("--max-lag", help="largest time lag fitted [frames], or 'auto': from --min-speed "
                                          "and --lag-pitches")
    parser.add_argument("--min-speed", type=float, help="slowest wind the lag range is sized for [m/s]")
    parser.add_argument("--lag-pitches", type=float, help="pitches a --min-speed layer moves within the lag range")
    parser.add_argument("--min-run-lags", type=float, help="skip closed-loop runs shorter than this many max lags")
    parser.add_argument("--n-layers", type=int, help="maximum number of layers")
    parser.add_argument("--min-peak", type=float, help="stop adding layers below this C_n^2")
    parser.add_argument("--low-order-modes", type=int, help="leading Z2C modes removed, from tip on")
    parser.add_argument("--n-zernike", type=int, help="low-pass the signal onto the first N Z2C modes (0: no filter)")
    parser.add_argument("--pupil-radius",
                        help="keep only slopes inside this radius [actuator pitches from the DM_Map centre]; "
                             "'auto': (Actuators_in_diameter - 1) / 2, 'none': every actuator")
    parser.add_argument("--max-speed",
                        help="bound the tracked peak's move per lag to this speed, per axis [m/s]; 'none': no bound")
    parser.add_argument("--transition-buffer", type=int, help="frames skipped after each open/closed-loop transition")
    parser.add_argument("--plot", nargs="?", const="last", choices=["last", "all"],
                        help="save PNGs for the last batch (default) or every batch")
    parser.add_argument("--save-cube", action="store_true", help="also save the correlation cubes")
    args = parser.parse_args()

    file_name = Path(args.file) if args.file else newest_file(output_dirs()[0])
    print(file_name)
    def number_or_word(value, words, kind):
        # "auto"/"none" pass through to profile_file, anything else is a number
        if value is None or value.lower() in words:
            return value
        return kind(value)

    try:
        profile = profile_file(file_name, signal=args.signal, frame_delay=args.frame_delay,
                               dm_sign=args.dm_sign, batch_size=args.batch_size,
                               max_lag=number_or_word(args.max_lag, ("auto",), int), min_speed=args.min_speed,
                               lag_pitches=args.lag_pitches, min_run_lags=args.min_run_lags,
                               n_layers=args.n_layers, min_peak=args.min_peak,
                               transition_buffer=args.transition_buffer, low_order_modes=args.low_order_modes,
                               n_zernike=args.n_zernike,
                               pupil_radius=number_or_word(args.pupil_radius, ("auto", "none"), float),
                               max_speed=number_or_word(args.max_speed, ("none",), float))
    except ProfilerInputError as e:
        sys.exit(str(e))
    save_results(file_name, profile, save_cube=args.save_cube)

    results = profile["results"]
    if not profile["fits"]:
        print("No closed-loop run long enough for one batch")
        return
    for k, t in enumerate(results["Iteration_Times"]):
        n = results["N_Layers"][k]
        layers = ", ".join(f"{s:.1f} m/s @ {d:.0f} deg ({c:.3f})" for s, d, c in
                           zip(results["Speed"][k][:n], results["Direction"][k][:n], results["Cn2_Fraction"][k][:n]))
        print(f"{datetime.fromtimestamp(t, timezone.utc):%H:%M:%S} UTC  V0 = {results['V0'][k]:.2f} m/s, "
              f"tau0 = {results['tau0'][k]:.2f} ms  |  {layers}")

    if args.plot:
        make_plots(profile, which=args.plot)


if __name__ == "__main__":
    main()
