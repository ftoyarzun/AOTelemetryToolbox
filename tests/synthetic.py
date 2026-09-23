"""
Small synthetic observation files for the tests, in the layout of
simulation/DataGeneration.ipynb.

A single Kolmogorov layer of known r0 blows at a known speed across a DM of
`n_across` x `n_across` actuators (pitch D / n_across, the frozen-flow
profiler's convention). A leaky integrator loop corrects it with one frame of
delay: meas(t) = phase(t) + dm(t-1), dm(t) = leak * dm(t-1) - gain * meas(t),
and the DM is held while the recorded loop status is 0. Z2C holds Noll
Zernikes sampled at the actuators, so its projection gives Zernike
coefficients in radians at 500 nm, the r0 reference wavelength. The science
frames are the diffraction-limited PSF of the annular pupil with a small
random jitter.
"""
from math import factorial

import h5py
import numpy as np
from scipy.ndimage import map_coordinates

WAVELENGTH_AO = 500e-9
WAVELENGTH_SCIENCE = 1550e-9
SAMPLING = 4
OBSTRUCTION = 0.3


def noll_nm(j):
    """Radial order n and azimuthal order m of Noll index j (j = 1 is piston)."""
    n = int((-1 + np.sqrt(8 * (j - 1) + 1)) / 2)
    p = j - n * (n + 1) // 2
    k = n % 2
    m = int((p + k) / 2) * 2 - k
    if m != 0:
        m *= 1 if j % 2 == 0 else -1
    return n, m


def zernike(j, rho, theta):
    """Noll-normalized Zernike polynomial j on the unit disk."""
    n, m = noll_nm(j)
    radial = sum((-1) ** k * factorial(n - k)
                 / (factorial(k) * factorial((n + abs(m)) // 2 - k) * factorial((n - abs(m)) // 2 - k))
                 * rho ** (n - 2 * k) for k in range((n - abs(m)) // 2 + 1))
    if m == 0:
        return np.sqrt(n + 1) * radial
    angular = np.cos(m * theta) if m > 0 else np.sin(-m * theta)
    return np.sqrt(2 * (n + 1)) * radial * angular


def kolmogorov_screen(shape, pixel, r0, rng):
    """Periodic Kolmogorov phase screen [rad], `pixel` metres per pixel, by FFT."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny, pixel)[:, None]
    fx = np.fft.fftfreq(nx, pixel)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = np.inf
    psd = 0.023 * r0 ** (-5 / 3) * f ** (-11 / 3)
    coefficients = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.sqrt(
        psd / (ny * pixel * nx * pixel))
    return np.real(np.fft.ifft2(coefficients)) * ny * nx


def annular_psf(size, pupil_px, tilt, obstruction):
    """PSF of an annular pupil `pupil_px` pixels across, on a `size` grid, with a
    (tip, tilt) wavefront slope in waves across the pupil."""
    c = np.arange(size) - size / 2
    x, y = np.meshgrid(c, c)
    r = np.hypot(x, y) / (pupil_px / 2)
    pupil = (r <= 1) & (r >= obstruction)
    phase = 2 * np.pi * (tilt[0] * x + tilt[1] * y) / pupil_px
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil * np.exp(1j * phase))))
    return np.abs(field) ** 2


def write_observation(path, duration=3.0, fps=1000, speed=5.0, direction_deg=30.0, r0=0.10, diameter=1.5,
                      n_across=17, loop_status=None, loop_status_in_attr=False, science=True,
                      science_fps=100, elevation=60.0, loop_gain=0.5, loop_leak=0.99, t0=1.7e9, seed=0):
    """
    Write a synthetic observation to `path` and return a dict of its true
    values (r0 at 500 nm in m, speed in m/s, tau0 in s, elevation).

    `loop_status` is the per-sample recorded loop command (default: closed
    loop throughout), written as the WFS/loop_status dataset, or as the WFS
    attribute with `loop_status_in_attr`, or left out when it is False.
    `science=False` leaves out the Science group.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration * fps))
    pitch = diameter / n_across

    # Actuators inside the pupil, on the n_across x n_across grid
    c = (np.arange(n_across) - (n_across - 1) / 2) * pitch
    ax0, ax1 = np.meshgrid(c, c, indexing="ij")
    dm_map = np.hypot(ax0, ax1) <= diameter / 2
    pos0, pos1 = ax0[dm_map], ax1[dm_map]
    n_act = pos0.size

    rho = np.hypot(pos0, pos1) / (diameter / 2)
    theta = np.arctan2(pos1, pos0)
    z2c = np.column_stack([zernike(j, rho, theta) for j in range(2, 52)])

    # The layer moves along (axis 0, axis 1) = (cos, sin) of direction_deg, the profiler's
    # convention: the phase at x and time t is the screen at x - v t
    pixel = pitch / 2
    screen = kolmogorov_screen((1024, 1024), pixel, r0, rng)
    direction = np.radians(direction_deg)
    t = np.arange(n_samples) / fps
    shift0 = speed * np.cos(direction) * t
    shift1 = speed * np.sin(direction) * t
    coords = [((pos0[None, :] - shift0[:, None]) / pixel).ravel(), ((pos1[None, :] - shift1[:, None]) / pixel).ravel()]
    phase = map_coordinates(screen, coords, order=1, mode="grid-wrap").reshape(n_samples, n_act)

    if loop_status is None:
        loop_status = np.ones(n_samples, dtype=int)
    closed = np.asarray(loop_status if loop_status is not False else np.ones(n_samples)) != 0
    dm = np.zeros((n_samples, n_act))
    meas = np.zeros((n_samples, n_act))
    previous = np.zeros(n_act)
    for k in range(n_samples):
        meas[k] = phase[k] + previous
        previous = loop_leak * previous - loop_gain * meas[k] if closed[k] else previous
        dm[k] = previous

    timestamps = t0 + t
    with h5py.File(path, "w") as file:
        file.attrs["Instrument"] = "Synthetic"
        file.attrs["Telescope"] = "Synthetic"

        wfs = file.create_group("WFS")
        wfs.attrs["Loop_Gain"] = loop_gain
        wfs.attrs["Loop_Leak"] = loop_leak
        wfs.attrs["Loop_Freq"] = fps
        if loop_status is not False:
            if loop_status_in_attr:
                wfs.attrs["loop_status"] = np.asarray(loop_status)
            else:
                wfs.create_dataset("loop_status", data=np.asarray(loop_status))
        images = wfs.create_dataset("WFS_Images", data=np.zeros((10, 8, 8)))
        images.attrs["Gain"] = 1
        images.attrs["FPS"] = fps
        images.attrs["Frame_Step"] = n_samples // 10
        wfs.create_dataset("WFS_TimeStamps", data=timestamps[::n_samples // 10][:10])
        wfs.create_dataset("Valid_Pixel_Map", data=np.ones((8, 8), dtype=bool))
        wfs.create_dataset("DM_commands", data=dm)
        wfs.create_dataset("DM_TimeStamps", data=timestamps)
        wfs.create_dataset("DM_Map", data=dm_map)
        wfs.create_dataset("WFS_measurements", data=meas)

        if science:
            sci = file.create_group("Science")
            sci.attrs["Target"] = "Synthetic star"
            sci.attrs["Elevation"] = elevation
            sci.attrs["Vmag"] = 3.0
            sci.attrs["Rmag"] = 2.8
            sci.attrs["Jmag"] = 2.5
            sci.attrs["Hmag"] = 2.4
            n_psf = int(round(duration * science_fps))
            size = 128
            frames = np.array([annular_psf(size, size // SAMPLING, rng.normal(0, 0.05, 2), OBSTRUCTION)
                               for _ in range(n_psf)])
            frames = frames / frames.max() * 1e4 + rng.normal(0, 1.0, frames.shape) + 10
            sci.create_dataset("PSF_TimeStamps", data=t0 + np.arange(n_psf) / science_fps)
            psfs = sci.create_dataset("Science_PSFs", data=frames)
            psfs.attrs["Exposure_Time"] = 1 / science_fps
            psfs.attrs["FPS"] = science_fps
            psfs.attrs["Gain"] = 1
            psfs.attrs["Sampling"] = SAMPLING
            psfs.attrs["Wavelength"] = WAVELENGTH_SCIENCE
            psfs.attrs["Bandpass"] = 0.0

        cal = file.create_group("Calibration")
        # One mode fewer than actuators, like OOPAO's (a square M2C has no telling orientation)
        cal.create_dataset("M2C", data=np.eye(n_act)[:, :n_act - 1])
        cal.create_dataset("Z2C", data=z2c)
        cal.attrs["Diameter"] = diameter
        cal.attrs["Obstruction_ratio"] = OBSTRUCTION
        cal.attrs["Science_Calibration_Wavelength"] = WAVELENGTH_SCIENCE
        cal.attrs["AO_Calibration_Wavelength"] = WAVELENGTH_AO
        cal.attrs["SkyCalibPupilRatio"] = 1.0
        cal.attrs["Actuators_in_diameter"] = n_across
        cal.attrs["Total_Number_Of_Actuators"] = n_act
        cal.attrs["Total_Number_Of_Controlled_Modes"] = n_act - 1
        cal.attrs["r0_reference_wvl"] = 500e-9

    return dict(r0=r0, speed=speed, direction_deg=direction_deg, tau0=0.314 * r0 / speed, elevation=elevation,
                science_fps=science_fps)
