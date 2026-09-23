"""
Standalone atmosphere/AO-loop characterization functions.

Every function here is self-contained: none depend on a class, on `self`
state, or on another function in this module having run first (parameters
a caller might otherwise pass implicitly via `self`, e.g. r0 into the tau0 estimate,
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
- estimate_wind_gain_delay_from_psd      Madec et al. 1992 / Conan et al.
                                          1995 (Zernike PSD cutoff-frequency
                                          law) + Poyneer et al. 2009 eq. 4
                                          (closed-loop PSD compensation)
- reconstruct_pseudo_open_loop           Fusco et al. 2004 eq. 3
- estimate_wind_speed_autocorrelation_cutoff
                                          Madec et al. 1992 / Fusco et al.
                                          2004 eq. 9-13, with the 1/e-width
                                          to cutoff-frequency constant of
                                          each radial order computed from
                                          Conan et al. 1995's frozen-flow
                                          model (autocorrelation_cutoff_constant)
- read_loop_status / find_status_runs    open/closed-loop status from the
                                          loop command recorded at every
                                          iteration (nonzero = closed loop)
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
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit
from scipy.signal import welch
from scipy.special import gamma, jv


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


def r0_at_zenith(r0, elevation_deg):
    """
    Line-of-sight r0 at `elevation_deg` converted to zenith: r0 scales as
    cos(z)^(3/5) with the zenith angle z, so r0_zenith = r0 * sin(elevation)^(-3/5).
    NaN where the elevation is NaN (unknown).
    """
    return np.asarray(r0, dtype=float) * np.sin(np.radians(elevation_deg)) ** (-3 / 5)


def tau0_at_zenith(tau0, elevation_deg):
    """
    Line-of-sight tau0 at `elevation_deg` converted to zenith. V0 is a
    normalized C_n^2 average of the wind speeds, so it does not depend on the
    elevation, and tau0 = 0.314 r0 / V0 scales like r0 (see r0_at_zenith).
    """
    return r0_at_zenith(tau0, elevation_deg)


def seeing_arcsec(r0, wavelength):
    """Seeing FWHM 0.98 wavelength / r0 [arcsec] of a Kolmogorov atmosphere, with r0
    and wavelength in the same length unit, r0 given at that wavelength."""
    return 0.98 * wavelength / np.asarray(r0, dtype=float) * 180 / np.pi * 3600


def seeing_at_zenith(seeing, elevation_deg):
    """Line-of-sight seeing at `elevation_deg` converted to zenith: seeing scales as
    1 / r0, so seeing_zenith = seeing * sin(elevation)^(3/5). NaN where the elevation is NaN."""
    return np.asarray(seeing, dtype=float) * np.sin(np.radians(elevation_deg)) ** (3 / 5)


# ---------------------------------------------------------------------------
# 2. Effective gain / effective delay -- full AO transfer-function fit to the
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
# 3. Pseudo-open-loop reconstruction -- Fusco et al. 2004 eq. 3
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
# 4. tau0/wind estimator -- autocorrelation cutoff (Madec et al. 1992 /
#    Fusco et al. 2004 eq. 9-13)
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


@lru_cache(maxsize=None)
def autocorrelation_cutoff_constant(n: int) -> float:
    """
    K_n = f_n * tau_n for Zernike radial order n under a single frozen-flow
    Kolmogorov layer of speed V, where f_n = 0.3 (n+1) V / D is the cutoff
    frequency of Conan et al. 1995 and tau_n the lag at which the order's
    temporal autocorrelation falls to 1/e.

    Summed over the modes of one radial order, the Zernike spectrum is
    isotropic, proportional to k^(-11/3) J_{n+1}(pi k D)^2 / k^2, so the
    autocorrelation at a shift x = V tau / D is

        A_n(x) = int u^(-14/3) J_{n+1}(u)^2 J_0(2 u x) du / int u^(-14/3) J_{n+1}(u)^2 du

    and K_n = 0.3 (n+1) x_e, with A_n(x_e) = 1/e: 0.316 for n = 2, then 0.269,
    0.254, 0.246, 0.242, 0.240, 0.238 for n = 3-8. The integrals are sums on a
    grid up to u = 100, accurate to 1e-4 in K_n.
    """
    u = np.linspace(1e-6, 100.0, 20001)
    weight = u ** (-14 / 3) * jv(n + 1, u) ** 2
    norm = trapezoid(weight, u)
    x_e = brentq(lambda x: trapezoid(weight * jv(0, 2 * u * x), u) / norm - np.exp(-1), 1e-4, 2.0, xtol=1e-8)
    return 0.3 * (n + 1) * x_e


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
    Wind speed (and, if r0 is given, tau0) from the 1/e width of each radial
    order's temporal autocorrelation (Madec et al. 1992 / Fusco et al. 2004
    eq. 9-13):
        f_n  = K_n / tau_n                              (cutoff frequency of order n)
        V0   = D * sum((n+1)*f_n) / sum(0.3*(n+1)^2)   (eq. 10-12)
        tau0 = 0.31 * r0 / V0                            (eq. 13, Roddier et al. 1982)

    tau_n is the lag where the order's autocorrelation, averaged over its
    modes, falls to 1/e, and K_n = autocorrelation_cutoff_constant(n), so that
    V0 is the layer speed for a single frozen-flow layer. The autocorrelation
    is taken over the batch after removing its mean, which shortens tau_n, and
    so raises V0, when the batch is not much longer than tau_n.
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

    cutoff_frequency = np.array([autocorrelation_cutoff_constant(int(n)) for n in orders]) / e_folding_lag

    valid = ~np.isnan(cutoff_frequency)
    n_plus_1 = orders[valid] + 1
    V0 = float(D * np.sum(n_plus_1 * cutoff_frequency[valid]) / np.sum(0.3 * n_plus_1 ** 2))

    tau0 = float(0.31 * r0 / V0) if r0 is not None else None

    return AutocorrelationWindResult(V0=V0, tau0=tau0, radial_orders=orders,
                                      cutoff_frequency=cutoff_frequency, e_folding_lag=e_folding_lag)


# ---------------------------------------------------------------------------
# 5. Open/closed-loop status, from the recorded loop command
# ---------------------------------------------------------------------------

def read_loop_status(wfs_grp) -> np.ndarray:
    """
    Per-sample closed-loop status (True = closed), one entry per row of
    `wfs_grp["DM_commands"]`: the loop command the RTC recorded at every
    loop iteration, any nonzero value meaning closed loop. It is the
    `loop_status` dataset of the open WFS group, or its `loop_status`
    attribute in simulated files.

    Raises KeyError if the file has neither, and ValueError if its length
    doesn't match DM_commands.
    """
    if "loop_status" in wfs_grp:
        loop_status = np.atleast_1d(wfs_grp["loop_status"][()])
    elif "loop_status" in wfs_grp.attrs:
        loop_status = np.asarray(wfs_grp.attrs["loop_status"])
    else:
        raise KeyError("no WFS/loop_status: the open/closed-loop status was not recorded")

    n_samples = wfs_grp["DM_commands"].shape[0]
    if loop_status.shape[0] != n_samples:
        raise ValueError(f"WFS/loop_status has {loop_status.shape[0]} samples, DM_commands has {n_samples}")
    return np.any(loop_status.reshape(n_samples, -1) != 0, axis=1)


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
# 6. DM-derived vs WFS-derived Zernike PSD -- a direct closed-loop diagnostic
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
# 7. Model-free loop bandwidth -- crossover of the DM- and WFS-derived PSDs
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
