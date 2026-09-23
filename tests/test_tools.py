"""Pure functions checked against known values."""
import h5py
import numpy as np
import pytest

from synthetic import OBSTRUCTION, SAMPLING, annular_psf

from aott.DataGrabber import SampleBuffer
from aott.PSF_Processing import frame_rate, strehl_ratio
from aott.atmosphere_characterization_tools import (
    autocorrelation_cutoff_constant,
    estimate_r0_L0,
    find_status_runs,
    r0_at_zenith,
    read_loop_status,
    seeing_arcsec,
    seeing_at_zenith,
    tau0_at_zenith,
    von_karman_zernike_radial_variance,
    zernike_radial_orders,
)


def test_autocorrelation_cutoff_constant():
    expected = [0.316, 0.269, 0.254, 0.246, 0.242, 0.240, 0.238]
    assert [autocorrelation_cutoff_constant(n) for n in range(2, 9)] == pytest.approx(expected, abs=1e-3)


def test_r0_at_zenith():
    assert r0_at_zenith(10.0, 90.0) == pytest.approx(10.0)
    # sin(30 deg)^(-3/5) = 2^(3/5)
    assert r0_at_zenith(10.0, 30.0) == pytest.approx(10.0 * 2 ** 0.6)
    assert np.isnan(r0_at_zenith(10.0, np.nan))


def test_tau0_and_seeing_at_zenith():
    assert tau0_at_zenith(2.0, 30.0) == pytest.approx(r0_at_zenith(2.0, 30.0))
    # 0.98 lambda / r0: 1.01 arcsec for r0 = 10 cm at 500 nm
    assert seeing_arcsec(0.1, 500e-9) == pytest.approx(1.0107, rel=1e-4)
    seeing = seeing_arcsec(0.1, 500e-9)
    assert seeing_at_zenith(seeing, 30.0) == pytest.approx(seeing_arcsec(r0_at_zenith(0.1, 30.0), 500e-9))


def test_estimate_r0_L0_recovers_r0():
    rng = np.random.default_rng(0)
    orders = zernike_radial_orders(50)
    variance = np.empty(50)
    fitted = orders >= 2
    variance[fitted] = von_karman_zernike_radial_variance(0.1, 25.0, 1.5, orders[fitted])
    variance[~fitted] = 1.0
    coefficients = rng.standard_normal((20000, 50)) * np.sqrt(variance)
    result = estimate_r0_L0(coefficients, 1.5, min_radial_order=3, max_radial_order=8)
    assert result.r0 == pytest.approx(0.1, rel=0.03)


def test_read_loop_status(tmp_path):
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        wfs = f.create_group("WFS")
        wfs.create_dataset("DM_commands", data=np.zeros((4, 3)))
        wfs.create_dataset("loop_status", data=[0, 1, 3, 0])
        assert read_loop_status(wfs).tolist() == [False, True, True, False]

        del wfs["loop_status"]
        wfs.attrs["loop_status"] = [1, 1, 0, 0]
        assert read_loop_status(wfs).tolist() == [True, True, False, False]

        wfs.attrs["loop_status"] = [1, 1, 0]
        with pytest.raises(ValueError):
            read_loop_status(wfs)

        del wfs.attrs["loop_status"]
        with pytest.raises(KeyError):
            read_loop_status(wfs)

        # One value for the whole file is not a per-iteration record
        wfs.create_dataset("loop_status", data=0)
        with pytest.raises(ValueError):
            read_loop_status(wfs)


def test_find_status_runs_drops_the_buffer_after_each_transition():
    status = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert find_status_runs(status, transition_buffer=1) == [(0, 3, True), (4, 6, False), (7, 10, True)]


def test_strehl_ratio_of_a_diffraction_limited_psf():
    psf = annular_psf(128, 128 // SAMPLING, (0.0, 0.0), OBSTRUCTION)
    psf = psf / psf.sum()
    assert strehl_ratio(psf, SAMPLING, OBSTRUCTION)[0] == pytest.approx(1.0, abs=1e-3)
    # Against an unobstructed reference, the same PSF reads low
    assert strehl_ratio(psf, SAMPLING)[0] < 0.95


def test_frame_rate_from_timestamps():
    assert frame_rate(1.7e9 + np.arange(100) / 50, 1000) == pytest.approx(50)
    assert frame_rate(np.array([1.7e9]), 1000) == 1000


def test_sample_buffer_grows_and_drops_length_one_axes():
    buffer = SampleBuffer(2)
    for k in range(5):
        buffer.append(np.full((1, 3), k))
    data = buffer.data()
    assert data.shape == (5, 3)
    assert data[:, 0].tolist() == [0, 1, 2, 3, 4]

    scalars = SampleBuffer(10)
    scalars.append(np.array([[1]]))
    assert scalars.data().shape == (1,)
