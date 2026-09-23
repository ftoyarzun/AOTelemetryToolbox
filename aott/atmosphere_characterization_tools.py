"""
Standalone atmosphere/AO-loop characterization functions.

Every function here is self-contained: none depend on a class, on `self`
state, or on another function in this module having run first (parameters
a caller might otherwise pass implicitly via `self`, e.g. r0 into the V0 fit,
are explicit required arguments here instead). `Atmosphere_Characterization`
(the class in Atmosphere_Characterization.py) is a thin orchestrator around
these functions -- it reads the HDF5 file, batches the telemetry, and calls
these; it does not reimplement any of this module's math itself.

Interface conventions
----------------------
- `zernike_modes` (and `dm_zernike_modes`/`wfs_zernike_modes`) are always 2D
  arrays shaped (n_samples, n_modes), one column per Zernike mode in
  increasing Noll-index order, starting at `first_noll_index` (default 2,
  i.e. column 0 = tip). Coefficients must already be in radians of phase,
  calibrated the same way `Atmosphere_Characterization` computes them
  (per-sample-mean-removed commands/measurements, projected through `C2Z`).
- `timestamps` is a 1D array shaped (n_samples,), in seconds. Sampling is
  assumed nearly uniform; the effective sample period is
  `np.median(np.diff(timestamps))`.
- No instrument constant (pupil diameter, obstruction ratio, wavelength, loop
  leak, ...) is hard-coded anywhere in this module -- every one is a required
  argument.

Paper -> function map (see papers/Literature_Comparison.md for the full
comparison)
------------------------------------------------------------------------
- estimate_r0_L0                        Fusco et al. 2004 (NAOS) eq. 8
- estimate_tau0_v0_structure_function    Kolmogorov D(tau); tau0/V0
                                          definitions match Berdeu et al.
                                          2025/2026 eq. 14 and SHIMM
                                          (Perera et al. 2023) eq. 5
- estimate_wind_gain_delay_from_psd      Madec et al. 1992 / Conan et al.
                                          1995 (Zernike PSD cutoff-frequency
                                          law) + Poyneer et al. 2009 eq. 4
                                          (closed-loop PSD compensation)
- reconstruct_pseudo_open_loop           Fusco et al. 2004 eq. 3
- estimate_wind_speed_autocorrelation_cutoff
                                          Madec et al. 1992 / Fusco et al.
                                          2004 eq. 9-13, cutoff-frequency
                                          constant recalibrated for this
                                          module's atmosphere PSD shape
                                          (see that function's docstring)
- detect_closed_loop_from_dm_commands / find_status_runs
                                          open/closed-loop status from DM
                                          command activity (dm[n] == dm[n+1]
                                          means open loop)
- compute_zernike_psd_comparison         per-mode DM-derived (pseudo
                                          open-loop atmosphere estimate) vs
                                          WFS-derived (closed-loop residual)
                                          PSD, for a caller-chosen list of
                                          modes
- estimate_loop_bandwidth_from_psd_ratio model-free per-radial-order loop
                                          bandwidth, from the crossover of
                                          the same two PSDs -- a cross-check
                                          on estimate_wind_gain_delay_from_psd's
                                          fitted gain/delay

Units of the structure function
-------------------------------
`zernike_structure_function` sums the squared per-mode coefficient
differences. Noll-normalized Zernike polynomials are orthonormal over the
pupil, so by Parseval this equals the pupil-averaged squared phase
difference, in rad^2, with no pupil-diameter factor (Conan 2008 eq. 17-19).
See "class reports/atmosphere_characterization_tools.md" for the numerical
check.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import welch
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Radial-order bookkeeping
# ---------------------------------------------------------------------------

def noll_radial_order(noll_index: int) -> int:
    """Standard Zernike radial order n of a Noll index j (j=1 -> n=0, j=2,3 -> n=1, ...)."""
    n = 0
    while (n + 1) * (n + 2) // 2 < noll_index:
        n += 1
    return n


def zernike_radial_orders(n_modes: int, first_noll_index: int = 2) -> np.ndarray:
    """Radial order of each column of a (n_samples, n_modes) Zernike array."""
    noll_indices = np.arange(first_noll_index, first_noll_index + n_modes)
    return np.array([noll_radial_order(j) for j in noll_indices])


def group_columns_by_radial_order(
    radial_orders_per_mode: np.ndarray,
    min_radial_order: Optional[int] = None,
    max_radial_order: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Map each present radial order n to the array indices (columns) belonging to it."""
    orders = sorted(set(radial_orders_per_mode.tolist()))
    if min_radial_order is not None:
        orders = [n for n in orders if n >= min_radial_order]
    if max_radial_order is not None:
        orders = [n for n in orders if n <= max_radial_order]
    return {n: np.where(radial_orders_per_mode == n)[0] for n in orders}


def _sample_period(timestamps: np.ndarray) -> float:
    return float(np.median(np.diff(timestamps)))


# ---------------------------------------------------------------------------
# 1. r0 / L0 -- Fusco et al. 2004 (NAOS) eq. 8
# ---------------------------------------------------------------------------

@dataclass
class R0L0Result:
    r0: float
    L0: float
    radial_orders: np.ndarray
    measured_variance: np.ndarray
    model_variance: np.ndarray


def von_karman_zernike_radial_variance(r0: float, L0: float, D: float, radial_orders: np.ndarray) -> np.ndarray:
    """
    Per-radial-order Zernike coefficient variance under von Karman statistics
    (Fusco et al. 2004, NAOS..., eq. 8). `radial_orders` are standard Noll
    radial orders n >= 2 (n=2 is defocus+astigmatism); tip/tilt (n=1) has no
    formula here and must be excluded upstream, as NAOS does.
    """
    radial_orders = np.asarray(radial_orders, dtype=float)
    if np.any(radial_orders < 2):
        raise ValueError("von_karman_zernike_radial_variance is only defined for radial order n >= 2")

    variance = np.empty_like(radial_orders)
    is_n2 = radial_orders == 2
    variance[is_n2] = 2.34e-2 * (D / r0) ** (5 / 3) * (
        1 - 0.39 * (2 * np.pi * D / L0) ** 2 + 0.27 * (2 * np.pi * D / L0) ** (7 / 3)
    )

    n = radial_orders[~is_n2]
    variance[~is_n2] = (
        0.756 * (n + 1) * gamma(n - 5 / 6) / gamma(n + 23 / 6) * (D / r0) ** (5 / 3)
        * (1 - 0.38 / ((n - 11 / 6) * (n + 23 / 6)) * (2 * np.pi * D / L0) ** 2)
    )
    return variance


def estimate_r0_L0(
    zernike_modes: np.ndarray,
    D: float,
    first_noll_index: int = 2,
    min_radial_order: int = 2,
    max_radial_order: Optional[int] = None,
    weight_range: tuple = (1.0, 10.0),
    r0_bounds: tuple = (1e-3, np.inf),
    L0_bounds: tuple = (1e-3, 40.0),
    initial_guess: tuple = (0.1, 25.0),
) -> R0L0Result:
    """
    Fit r0 and L0 to the measured per-radial-order Zernike variance. Excludes
    tip/tilt by default (min_radial_order=2), matching NAOS/Fusco et al. 2004
    section 3.3 ("Tip-tilt coefficients are excluded because of the possible
    additional errors caused by telescope vibrations and tracking errors").

    Each residual is divided by a linear `weight_range` scale across radial
    orders, which down-weights the low orders.

    `L0_bounds` caps L0 at 40 m: with only a few radial orders to constrain
    it, an unbounded fit can diverge to unphysical values.
    """
    n_samples, n_modes = zernike_modes.shape
    radial_orders_per_mode = zernike_radial_orders(n_modes, first_noll_index)
    groups = group_columns_by_radial_order(radial_orders_per_mode, min_radial_order, max_radial_order)
    orders = np.array(sorted(groups))

    per_mode_variance = np.var(zernike_modes, axis=0)
    measured_variance = np.array([per_mode_variance[groups[n]].mean() for n in orders])

    weight = np.linspace(weight_range[0], weight_range[1], len(orders))

    def model(_, r0, L0):
        return von_karman_zernike_radial_variance(r0, L0, D, orders)

    fit, _ = curve_fit(
        model, orders, measured_variance,
        p0=initial_guess, bounds=([r0_bounds[0], L0_bounds[0]], [r0_bounds[1], L0_bounds[1]]),
    )
    r0_fit, L0_fit = fit
    model_variance = von_karman_zernike_radial_variance(r0_fit, L0_fit, D, orders)

    return R0L0Result(r0=r0_fit, L0=L0_fit, radial_orders=orders,
                       measured_variance=measured_variance, model_variance=model_variance)


# ---------------------------------------------------------------------------
# 2. tau0 / V0 -- temporal structure function of the Zernike-projected phase
# ---------------------------------------------------------------------------

@dataclass
class Tau0V0Result:
    tau0: float
    V0: float
    lags: np.ndarray
    structure_function: np.ndarray
    crossed_threshold: bool
    max_lag_reached: int


def zernike_structure_function(zernike_modes: np.ndarray, timestamps: np.ndarray, max_lag: int):
    """
    D(tau) = <|phi(t+tau) - phi(t)|^2>, computed as the sum over modes of the
    per-mode mean squared coefficient difference at each lag (see the module
    docstring for the units).
    """
    n_samples = zernike_modes.shape[0]
    max_lag = min(max_lag, n_samples - 1)
    lags = np.arange(max_lag + 1)
    D_tau = np.empty(max_lag + 1)
    for k in lags:
        if k == 0:
            D_tau[k] = 0.0
        else:
            diff = zernike_modes[:-k] - zernike_modes[k:]
            D_tau[k] = np.mean(np.sum(diff ** 2, axis=1))
    dt = _sample_period(timestamps)
    return lags * dt, D_tau


def estimate_tau0_v0_structure_function(
    zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    r0: float,
    initial_max_lag: int = 20,
    growth_factor: float = 2.0,
    max_lag_fraction: float = 0.25,
    crossing_level: float = 1.0,
    safety_multiplier: float = 2.0,
    v0_bounds: tuple = (1e-3, np.inf),
    v0_initial_guess: float = 10.0,
) -> Tau0V0Result:
    """
    tau0: lag at which D(tau) first crosses `crossing_level` (1 rad^2).
    V0: fit of the Kolmogorov structure function 6.88*(V*tau/r0)^(5/3) to D(tau).

    The lag-search window grows geometrically (doubling by default) from
    `initial_max_lag`, capped at `max_lag_fraction` of the batch length. If
    D(tau) never reaches the crossing, tau0 and V0 are NaN and
    crossed_threshold is False.
    """
    n_samples = zernike_modes.shape[0]
    hard_cap = max(int(n_samples * max_lag_fraction), initial_max_lag)

    max_lag = min(initial_max_lag, hard_cap)
    lag_times, D_tau = zernike_structure_function(zernike_modes, timestamps, max_lag)

    while D_tau.max() < safety_multiplier * crossing_level and max_lag < hard_cap:
        new_max_lag = min(int(max_lag * growth_factor), hard_cap)
        if new_max_lag == max_lag:
            break
        max_lag = new_max_lag
        lag_times, D_tau = zernike_structure_function(zernike_modes, timestamps, max_lag)

    crossed = bool(D_tau.max() >= crossing_level)
    if not crossed:
        return Tau0V0Result(tau0=np.nan, V0=np.nan, lags=lag_times, structure_function=D_tau,
                             crossed_threshold=False, max_lag_reached=max_lag)

    over = np.where(D_tau >= crossing_level)[0]
    i1 = over[0]
    if i1 == 0:
        tau0 = float(lag_times[0])
    else:
        interpolator = interp1d(D_tau[i1 - 1:i1 + 1], lag_times[i1 - 1:i1 + 1], kind="linear")
        tau0 = float(interpolator(crossing_level))

    def model(delay, wind_speed):
        return 6.88 * (wind_speed * delay / r0) ** (5 / 3)

    V0_fit, _ = curve_fit(model, lag_times, D_tau, p0=[v0_initial_guess], bounds=v0_bounds)

    return Tau0V0Result(tau0=tau0, V0=float(V0_fit[0]), lags=lag_times, structure_function=D_tau,
                         crossed_threshold=True, max_lag_reached=max_lag)


# ---------------------------------------------------------------------------
# 3. Effective gain / effective delay -- full AO transfer-function fit to the
#    per-radial-order temporal PSD
# ---------------------------------------------------------------------------

@dataclass
class WindGainDelayResult:
    effective_gain: float
    effective_delay: float
    radial_orders: np.ndarray
    frequency: np.ndarray
    psd: np.ndarray
    fitted_psd: np.ndarray
    per_order_params: Dict[str, np.ndarray]


def _ao_loop_transfer_functions(f, period, ki, leak, n_frames_delay):
    s = 2j * np.pi * f
    sT = s * period
    H_hold = (1 - np.exp(-sT)) / sT  # zero-order hold, shared by DM and WFS
    H_delay = np.exp(-n_frames_delay * sT)
    H_controller = ki / (leak - np.exp(-sT))  # leaky integrator
    return H_hold, H_delay, H_controller


def _low_pass(f, omega, alpha):
    return 1.0 / (1.0 + (f / omega) ** alpha)


def _closed_loop_psd_model(f, period, amp, alpha, omega, alpha1, ki, leak, n_frames_delay, noise):
    """Poyneer et al. 2009 eq. 4: closed-loop DM-command PSD implied by an
    atmosphere PSD model, the loop's rejection transfer function, and a flat
    WFS noise floor."""
    f_safe = f + f[1]  # avoid the f=0 singularity in the power-law atmosphere term
    H_hold, H_delay, H_controller = _ao_loop_transfer_functions(f_safe, period, ki, leak, n_frames_delay)
    loop = H_hold * H_delay * H_controller
    rejection = np.abs(loop / (1 + H_hold * loop)) ** 2
    atmosphere_psd = amp * f_safe ** (-alpha) * _low_pass(f, omega, alpha1) * np.abs(H_hold) ** 2
    return rejection * (atmosphere_psd + noise)


def _fit_radial_order_psd(f, psd, period, leak, weight_range, bounds, p0, maxfev):
    weight = np.linspace(weight_range[0], weight_range[1], f.shape[0])

    def model(_, amp, alpha, omega, alpha1, ki, n_frames_delay, noise):
        return np.log(_closed_loop_psd_model(f, period, amp, alpha, omega, alpha1, ki, leak, n_frames_delay, noise)) / weight

    fit, _ = curve_fit(model, f, np.log(psd) / weight, p0=p0, bounds=bounds, maxfev=maxfev)
    return fit


def estimate_wind_gain_delay_from_psd(
    zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    leak: float,
    first_noll_index: int = 2,
    min_radial_order: int = 2,
    max_radial_order: Optional[int] = None,
    nperseg: int = 1000,
    ki_bounds: tuple = (0.1, 0.8),
    delay_bounds: tuple = (1.0, 3.5),
    alpha1_bounds: tuple = (5 / 3, 23 / 3),
    alpha_bounds: tuple = (0.0, 1.0),
    amp_bounds: tuple = (0.0, np.inf),
    omega_bounds: tuple = (0.0, 200.0),
    noise_bounds: tuple = (0.0, 1.0),
    initial_guess: tuple = (5e-1, 0.5, 5.0, 11 / 3, 0.44, 1.8, 0.5e-6),
    weight_range: tuple = (1.0, 10.0),
    maxfev: int = 1500,
) -> WindGainDelayResult:
    """
    Fits _closed_loop_psd_model to each radial order's temporal PSD, with
    `leak` fixed to the telemetry's known loop leak (not fitted), and
    combines the per-order fitted gain/delay into one effective value each
    (plain mean over radial order).

    Only meaningful for a batch drawn entirely from closed-loop telemetry,
    with the DM-derived Zernike modes as input: the model assumes an active
    leaky-integrator loop.

    The per-order fitted cutoff frequency is returned in
    `per_order_params["cutoff_frequency"]` but not combined into a wind
    speed: above radial order ~5 the atmospheric knee is too weak to separate
    from the loop's own rejection shaping, and the combination weights those
    orders most (see "class reports/atmosphere_characterization_tools.md").

    Returns both the aggregated gain/delay and the full per-radial-order fit
    (`per_order_params`) for diagnostics.
    """
    n_samples, n_modes = zernike_modes.shape
    dt = _sample_period(timestamps)
    fs = 1.0 / dt
    period = dt

    radial_orders_per_mode = zernike_radial_orders(n_modes, first_noll_index)
    groups = group_columns_by_radial_order(radial_orders_per_mode, min_radial_order, max_radial_order)
    orders = np.array(sorted(groups))

    f, psd_per_mode = welch(zernike_modes.T, fs, nperseg=nperseg)
    psd_per_order = np.array([psd_per_mode[groups[n]].mean(axis=0) for n in orders])

    positive = f > 0
    f_fit = f[positive]
    psd_fit = psd_per_order[:, positive]

    lower = [amp_bounds[0], alpha_bounds[0], omega_bounds[0], alpha1_bounds[0], ki_bounds[0], delay_bounds[0], noise_bounds[0]]
    upper = [amp_bounds[1], alpha_bounds[1], omega_bounds[1], alpha1_bounds[1], ki_bounds[1], delay_bounds[1], noise_bounds[1]]

    n_orders = len(orders)
    per_order_fit = np.empty((n_orders, 7))
    fitted_psd = np.empty_like(psd_fit)
    for i in range(n_orders):
        per_order_fit[i] = _fit_radial_order_psd(f_fit, psd_fit[i], period, leak, weight_range, (lower, upper), initial_guess, maxfev)
        amp, alpha, omega, alpha1, ki, n_frames_delay, noise = per_order_fit[i]
        fitted_psd[i] = _closed_loop_psd_model(f_fit, period, amp, alpha, omega, alpha1, ki, leak, n_frames_delay, noise)

    omega = per_order_fit[:, 2]
    ki = per_order_fit[:, 4]
    n_frames_delay = per_order_fit[:, 5]

    effective_gain = float(np.mean(ki))
    effective_delay = float(np.mean(n_frames_delay))

    return WindGainDelayResult(
        effective_gain=effective_gain, effective_delay=effective_delay,
        radial_orders=orders, frequency=f_fit, psd=psd_fit, fitted_psd=fitted_psd,
        per_order_params={
            "amplitude": per_order_fit[:, 0], "alpha": per_order_fit[:, 1],
            "cutoff_frequency": omega, "alpha1": per_order_fit[:, 3],
            "gain": ki, "delay_frames": n_frames_delay, "noise": per_order_fit[:, 6],
        },
    )


# ---------------------------------------------------------------------------
# 4. Pseudo-open-loop reconstruction -- Fusco et al. 2004 eq. 3
# ---------------------------------------------------------------------------

def reconstruct_pseudo_open_loop(
    dm_zernike_modes: np.ndarray,
    wfs_zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    frame_delay: int = 2,
):
    """
    a_rec(t) = a_volt(t) + a_slope(t + frame_delay)  (Fusco et al. 2004 eq. 3).

    `dm_zernike_modes` and `wfs_zernike_modes` must be the DM-command-derived
    and WFS-measurement-derived Zernike coefficients for the SAME time
    samples (same shape, same `timestamps`). `frame_delay` is the AO loop's
    read-out/compute delay in frames -- supply your own loop's value, it is
    not assumed.

    Equivalent in spirit to Berdeu et al. 2025/2026's pseudo-open-loop slopes
    (residual slopes + the DM-command contribution reprojected through the
    interaction matrix), just expressed in modal (Zernike) rather than
    per-subaperture form.

    Returns (reconstructed_modes, trimmed_timestamps); the last `frame_delay`
    samples have no valid WFS(t+delay) partner and are dropped.
    """
    if dm_zernike_modes.shape != wfs_zernike_modes.shape:
        raise ValueError("dm_zernike_modes and wfs_zernike_modes must have the same shape")

    n_samples = dm_zernike_modes.shape[0]
    if frame_delay >= n_samples:
        raise ValueError("frame_delay must be smaller than the number of samples")

    valid = n_samples - frame_delay
    reconstructed = dm_zernike_modes[:valid] + wfs_zernike_modes[frame_delay:frame_delay + valid]
    return reconstructed, timestamps[:valid]


# ---------------------------------------------------------------------------
# 5. Second tau0/wind estimator -- autocorrelation-cutoff (Madec et al. 1992 /
#    Fusco et al. 2004 eq. 9-13), independent of the structure-function one
# ---------------------------------------------------------------------------

@dataclass
class AutocorrelationWindResult:
    V0: float
    tau0: Optional[float]
    radial_orders: np.ndarray
    cutoff_frequency: np.ndarray
    e_folding_lag: np.ndarray


def _temporal_autocorrelation(zernike_modes: np.ndarray, max_lag: int):
    centered = zernike_modes - zernike_modes.mean(axis=0, keepdims=True)
    lags = np.arange(max_lag + 1)
    autocorr = np.empty((max_lag + 1, zernike_modes.shape[1]))
    for k in lags:
        if k == 0:
            autocorr[k] = np.mean(centered * centered, axis=0)
        else:
            autocorr[k] = np.mean(centered[:-k] * centered[k:], axis=0)
    return lags, autocorr


def estimate_wind_speed_autocorrelation_cutoff(
    zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    D: float,
    r0: Optional[float] = None,
    first_noll_index: int = 2,
    min_radial_order: int = 2,
    max_radial_order: Optional[int] = None,
    n_lags: Optional[int] = None,
) -> AutocorrelationWindResult:
    """
    Independent wind-speed (and, if r0 is given, tau0) estimator from the 1/e
    width of each radial order's temporal autocorrelation, for cross-checking
    against estimate_tau0_v0_structure_function's structure-function-based
    value.

    Loosely follows Madec et al. 1992 / Fusco et al. 2004 eq. 9-13 (itself
    calibrated against Conan et al. 1995's exact theoretical Zernike temporal
    PSD shape):
        V0   = D * sum((n+1)*f_n) / sum(0.3*(n+1)^2)   (eq. 10-12)
        tau0 = 0.31 * r0 / V0                            (eq. 13, Roddier et al. 1982)

    The cutoff-frequency step (eq. 9) uses

        f_n = 1 / (2*pi * 1.15 * tau_n_1_over_e)

    instead of Fusco et al. 2004's `f_n = 1.15*pi / tau_n_1_over_e`. Their
    constant is calibrated against Conan et al. 1995's exact Zernike PSD
    shape, and overestimates f_n by ~11.7x for the single-knee shape used in
    this module (`_low_pass(f, f_c, alpha1)`, see `_closed_loop_psd_model`).
    See "class reports/atmosphere_characterization_tools.md".
    """
    n_samples, n_modes = zernike_modes.shape
    dt = _sample_period(timestamps)
    if n_lags is None:
        n_lags = min(n_samples // 4, 500)

    radial_orders_per_mode = zernike_radial_orders(n_modes, first_noll_index)
    groups = group_columns_by_radial_order(radial_orders_per_mode, min_radial_order, max_radial_order)
    orders = np.array(sorted(groups))

    lags_idx, autocorr = _temporal_autocorrelation(zernike_modes, n_lags)
    lag_times = lags_idx * dt

    e_folding_lag = np.full(len(orders), np.nan)
    for k, n in enumerate(orders):
        mode_autocorr = autocorr[:, groups[n]].mean(axis=1)
        if mode_autocorr[0] <= 0:
            continue
        normalized = mode_autocorr / mode_autocorr[0]
        below = np.where(normalized <= 1 / np.e)[0]
        if len(below) == 0:
            continue
        i1 = below[0]
        if i1 == 0:
            e_folding_lag[k] = lag_times[0]
        else:
            interpolator = interp1d(normalized[i1 - 1:i1 + 1], lag_times[i1 - 1:i1 + 1], kind="linear")
            e_folding_lag[k] = float(interpolator(1 / np.e))

    cutoff_frequency = 1 / e_folding_lag / (1.15 * 2 * np.pi)  # recalibrated eq. 9, see docstring above

    valid = ~np.isnan(cutoff_frequency)
    n_plus_1 = orders[valid] + 1
    V0 = float(D * np.sum(n_plus_1 * cutoff_frequency[valid]) / np.sum(0.3 * n_plus_1 ** 2))

    tau0 = float(0.31 * r0 / V0) if r0 is not None else None

    return AutocorrelationWindResult(V0=V0, tau0=tau0, radial_orders=orders,
                                      cutoff_frequency=cutoff_frequency, e_folding_lag=e_folding_lag)


# ---------------------------------------------------------------------------
# 6. Open/closed-loop status, derived from DM command activity
# ---------------------------------------------------------------------------

def detect_closed_loop_from_dm_commands(dm_commands: np.ndarray) -> np.ndarray:
    """
    Per-sample closed-loop status derived directly from DM command activity:
    the loop is closed for sample n if the DM command changes between sample
    n and n+1 (dm[n] != dm[n+1] for at least one actuator); it is open if the
    DM is held static.

    Returns a bool array the same length as `dm_commands` (True = closed
    loop). The last sample copies the previous sample's status, since there
    is no n+1 to compare it against.
    """
    n_samples = dm_commands.shape[0]
    if n_samples < 2:
        return np.ones(n_samples, dtype=bool)
    changed = np.any(dm_commands[1:] != dm_commands[:-1], axis=1)
    is_closed = np.empty(n_samples, dtype=bool)
    is_closed[:-1] = changed
    is_closed[-1] = changed[-1]
    return is_closed


def find_status_runs(status: np.ndarray, transition_buffer: int = 0):
    """
    Contiguous runs of a boolean (or other) status array, as a list of
    `(start, end, status)` triples (`end` exclusive). Used to walk the
    closed- and open-loop stretches of a file one run at a time, without
    ever mixing samples from two different runs into the same analysis batch.

    `transition_buffer` drops that many samples from the start of every run
    that follows a status change, since right after the loop closes (or
    opens) the system has not settled into the new regime yet -- e.g. a
    freshly closed loop still looks open-loop until it converges. The first
    run of the array is kept whole (there is no transition before it), and a
    run no longer than the buffer is dropped entirely.
    """
    status = np.asarray(status)
    changes = np.where(np.diff(status.astype(int)) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(status)]))
    runs = []
    for s, e in zip(starts, ends):
        trimmed_start = s + transition_buffer if s > 0 else s
        if trimmed_start < e:
            runs.append((int(trimmed_start), int(e), status[s]))
    return runs


# ---------------------------------------------------------------------------
# 7. DM-derived vs WFS-derived Zernike PSD -- a direct closed-loop diagnostic
# ---------------------------------------------------------------------------

@dataclass
class ZernikePSDComparisonResult:
    modes: np.ndarray
    frequency: np.ndarray
    dm_psd: np.ndarray   # (n_modes, n_freq)
    wfs_psd: np.ndarray  # (n_modes, n_freq)


@dataclass
class ZernikePSDResult:
    modes: np.ndarray
    frequency: np.ndarray
    psd: np.ndarray  # (n_modes, n_freq)


def compute_zernike_psd(
    zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    modes,
    nperseg: int = 500,
) -> ZernikePSDResult:
    """
    Per-mode temporal PSD of a Zernike-coefficient time series, for a
    caller-chosen list of mode indices (0-based, e.g. [0, 1, 2, 3]). No
    assumption about loop status: this is the right call for an open-loop
    batch, where the WFS-derived Zernike modes directly measure the
    atmosphere (no DM correction to compare against, unlike
    compute_zernike_psd_comparison's closed-loop DM-vs-WFS pairing).
    """
    modes = np.asarray(modes)
    dt = _sample_period(timestamps)
    fs = 1.0 / dt
    f, psd = welch(zernike_modes[:, modes].T, fs, nperseg=nperseg)
    return ZernikePSDResult(modes=modes, frequency=f, psd=psd)


def compute_zernike_psd_comparison(
    dm_zernike_modes: np.ndarray,
    wfs_zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    modes,
    nperseg: int = 500,
) -> ZernikePSDComparisonResult:
    """
    Per-mode temporal PSD of the DM-derived Zernike coefficients (an estimate
    of the open-loop/uncorrected atmosphere, since in closed loop the DM shape
    tracks the atmosphere) alongside the WFS-derived Zernike coefficients (the loop's actual
    real-time closed-loop residual), for a caller-chosen list of mode indices
    (0-based, into the zernike_modes column ordering, e.g. [0, 1, 2, 3]).

    Only meaningful when the loop is closed for the whole batch: in open loop
    the DM is static and its PSD carries no information -- use
    compute_zernike_psd on the WFS-derived modes instead.
    """
    modes = np.asarray(modes)
    dm_result = compute_zernike_psd(dm_zernike_modes, timestamps, modes, nperseg=nperseg)
    wfs_result = compute_zernike_psd(wfs_zernike_modes, timestamps, modes, nperseg=nperseg)
    return ZernikePSDComparisonResult(modes=modes, frequency=dm_result.frequency,
                                       dm_psd=dm_result.psd, wfs_psd=wfs_result.psd)


# ---------------------------------------------------------------------------
# 8. Model-free loop bandwidth -- crossover of the DM- and WFS-derived PSDs
# ---------------------------------------------------------------------------

@dataclass
class LoopBandwidthResult:
    radial_orders: np.ndarray
    frequency: np.ndarray
    dm_psd: np.ndarray
    wfs_psd: np.ndarray
    crossover_frequency: np.ndarray


def estimate_loop_bandwidth_from_psd_ratio(
    dm_zernike_modes: np.ndarray,
    wfs_zernike_modes: np.ndarray,
    timestamps: np.ndarray,
    first_noll_index: int = 2,
    min_radial_order: int = 2,
    max_radial_order: Optional[int] = None,
    nperseg: int = 500,
) -> LoopBandwidthResult:
    """
    Model-free per-radial-order loop bandwidth: the frequency at which the
    WFS-measured (closed-loop residual) PSD first rises above the DM-derived
    (pseudo open-loop atmosphere) PSD, i.e. where the loop stops keeping up
    with the turbulence. Independent of estimate_wind_gain_delay_from_psd's
    parametric transfer-function fit -- a direct cross-check on its fitted
    effective_gain/effective_delay, from the same two PSDs
    compute_zernike_psd_comparison plots. Only meaningful for closed-loop
    batches.

    `crossover_frequency[i]` is NaN for a radial order whose WFS PSD never
    exceeds its DM PSD over the fitted frequency range.
    """
    n_modes = dm_zernike_modes.shape[1]
    dt = _sample_period(timestamps)
    fs = 1.0 / dt

    radial_orders_per_mode = zernike_radial_orders(n_modes, first_noll_index)
    groups = group_columns_by_radial_order(radial_orders_per_mode, min_radial_order, max_radial_order)
    orders = np.array(sorted(groups))

    f, dm_psd_per_mode = welch(dm_zernike_modes.T, fs, nperseg=nperseg)
    _, wfs_psd_per_mode = welch(wfs_zernike_modes.T, fs, nperseg=nperseg)
    dm_psd = np.array([dm_psd_per_mode[groups[n]].mean(axis=0) for n in orders])
    wfs_psd = np.array([wfs_psd_per_mode[groups[n]].mean(axis=0) for n in orders])

    positive = f > 0
    f_pos = f[positive]
    dm_psd = dm_psd[:, positive]
    wfs_psd = wfs_psd[:, positive]

    crossover_frequency = np.full(len(orders), np.nan)
    for i in range(len(orders)):
        ratio = wfs_psd[i] / dm_psd[i]
        above = np.where(ratio >= 1.0)[0]
        if len(above) == 0:
            continue
        idx = above[0]
        if idx == 0:
            crossover_frequency[i] = f_pos[0]
        else:
            interpolator = interp1d(np.log(ratio[idx - 1:idx + 1]), f_pos[idx - 1:idx + 1], kind="linear")
            crossover_frequency[i] = float(interpolator(0.0))

    return LoopBandwidthResult(radial_orders=orders, frequency=f_pos, dm_psd=dm_psd, wfs_psd=wfs_psd,
                                crossover_frequency=crossover_frequency)
