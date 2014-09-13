from brownian import (BrownianMotionFitter, u, calc_k_c,
                      translate_fit_parameters, avg_ci_data, get_data,
                      fit_residuals, silentremove)
import h5py
import numpy as np
from nose.tools import assert_almost_equal
import unittest
from uncertainties import ufloat
from numpy.testing import assert_array_almost_equal
from jittermodel.base import Cantilever

# Load some test data from an hdf5 file.
# TODO: Get rid of dependence on .h5 file; this data can be programatically
#       generated.
with h5py.File('data.h5', 'r') as data:
    f = data['f'].value
    PSDx = data['PSDx'].value
    PSD_wgt = data['PSDw'].value

est_cant = Cantilever(f_c=63700*u.Hz, k_c=3.5*u.N/u.m,
                      Q=20000*u.dimensionless)
T = 295


class testBrownianMotionFitter_init(unittest.TestCase):
    """Test the initialization of the BrownianMotionFitter"""
    def setUp(self):
        self.reduced_f = f[::25000]
        self.reduced_PSD = PSDx[::25000]
        self.reduced_PSD_wgt = self.reduced_PSD * 0.25

    def test_BrownMotionFitter_init_success(self):
        # Try a case that should work.

        a = BrownianMotionFitter(self.reduced_f, self.reduced_PSD,
                                 self.reduced_PSD_wgt, T, est_cant)
        assert np.all(a.f == self.reduced_f)
        assert np.all(a.PSD_raw == self.reduced_PSD)
        assert np.all(a.T == T * u.K)
        assert a.est_cant == est_cant


ex_scaled_PSD = np.load('scale_data_PSD.npy')


class testBrownianMotionFitter_fitting(unittest.TestCase):
    def setUp(self):
        self.bmf = BrownianMotionFitter(f, PSDx, PSD_wgt, T, est_cant)

    def test_guess_P_detector(self):
        self.bmf._guess_P_detector()
        assert_almost_equal(self.bmf.P_detector0_raw, 6.5209499999999999e-09)

    def test_scale_data(self):
        self.bmf.P_detector0_raw = 6.5209499999999999e-09
        self.bmf._scale_data()
        np.testing.assert_allclose(ex_scaled_PSD, self.bmf.PSD)
        assert self.bmf.P_detector0 == 1

    def test_calc_initial_params(self):
        self.bmf.P_detector0_raw = 6.5209499999999999e-09
        self.bmf.P_detector0 = 1
        self.bmf.PSD = ex_scaled_PSD
        self.bmf._calc_initial_params()

        ex_initial_params = [8.9173850434295625e-05, 63700, 20000]

        np.testing.assert_allclose(ex_initial_params, self.bmf.initial_params)


def test_calc_k_c():
    """Test error calculation for calc_k_c.

    The calculation should use the (relative errors)

    P_x0: 0.01
    f_c: 1e-5
    Q: 0.01
    T: 0.01"""

    P_x0 = ufloat(1.75789868673e-12, 1.75789868673e-14) * u.nm**2/u.Hz  # 1/100
    f_c = ufloat(50000, 0.5) * u.Hz  # 1/100000 relative
    Q = ufloat(10000, 100) * u.dimensionless  # 1/100
    T = ufloat(300, 3) * u.K  # 1/100
    ex_k_c = ufloat(3, 0.05196153288731964) * u.N/u.m
    k_c = calc_k_c(f_c, Q, P_x0, T)
    assert_almost_equal(k_c.magnitude.n, ex_k_c.magnitude.n)
    assert_almost_equal(k_c.magnitude.s, ex_k_c.magnitude.s)


def test_translate_fit_parameters():
    popt = np.array([1.75789868673e-6, 50e3, 10000, 1e-6])
    pcov = np.array([[1.75789868673e-8**2, 0, 0, 0],
                    [0, 0.5**2, 0, 0],
                    [0, 0, 100**2, 0],
                    [0, 0, 0, 1e-7**2]])
    P_detector0_raw = 1e-6
    T = ufloat(300, 3) * u.K
    ex_output = [ufloat(50000, 0.5) * u.Hz,
                 ufloat(3, 0.05196153288731964) * u.N/u.m,
                 ufloat(10000, 100) * u.dimensionless,
                 ufloat(1e-12, 1e-13) * u.nm ** 2 / u.Hz]
    output = translate_fit_parameters(popt, pcov, P_detector0_raw, T)
    for ex_i, i in zip(ex_output, output):
        assert_almost_equal(ex_i.magnitude.n, i.magnitude.n)
        assert_almost_equal(ex_i.magnitude.s, i.magnitude.s)


def test_avg_ci_data():
    data = np.array([[1.1, 2.4, 3.95, 5.],
                     [1.2, 2.7, 3.9, 5.4],
                     [1.2, 2.4, 4., 5.7],
                     [1.3, 2.7, 3.8, 4.7]])
    ex_avg = np.array([1.2, 2.55, 3.9125, 5.2])
    ex_ci = np.array([0.08001666, 0.16974098, 0.08368343, 0.43090293])
    avg, ci = avg_ci_data(data)
    assert_array_almost_equal(ex_avg, avg)
    assert_array_almost_equal(ex_ci, ci)


def test_get_data():
    ex_f = np.array([1, 2, 3, 4, 5])
    ex_PSD = np.array([[1.1,  1.2, 1.2, 1.3],
                       [2.4,  2.7, 2.4, 2.7],
                       [3.95, 3.9, 4,   3.8],
                       [5,    5.4, 5.7, 4.7],
                       [5,    6,   6,   4]])

    # Create the test HDF5 file
    with h5py.File('test.h5', 'w') as fh:
        fh['x'] = ex_f
        fh['y'] = ex_PSD.mean(axis=1)
        fh['y'].attrs['n_avg'] = ex_PSD.shape[1]
        # Need ddof=1 to get sample standard deviation, rather than population.
        fh['y_std'] = ex_PSD.std(axis=1, ddof=1)

    ex_mean = np.array([1.2, 2.55, 3.9125, 5.2, 5.25])
    ex_ci = np.array([0.080016664930917122, 0.16974097914175013,
                      0.083683431255336824, 0.43090292797024871,
                      0.93827856560121137])

    f, psd_mean, psd_ci = get_data('test.h5')

    assert np.allclose(ex_f, f)
    assert np.allclose(ex_mean, psd_mean)
    assert np.allclose(ex_ci, psd_ci)


class TestOldDataFormat(unittest.TestCase):
    """Test the old data format, with data stored in a group 'PSD', and
    individual power spectra stored underneath that as
    'PSD/d001', 'PSD/d002'..."""
    def setUp(self):
        # Filenames
        self.testv1 = 'test-data-v1.h5'  # Old data format
        self.testv2 = 'test-data-v2.h5'
        self.converted = 'converted-data.h5'
        # Sample data
        self.ex_f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.d1 = np.array([1.1, 2.4, 3.95, 5.,  5.])
        self.d2 = np.array([1.2, 2.7, 3.9, 5.4,  6.])
        self.d3 = np.array([1.2, 2.4, 4., 5.7,  6.])
        self.d4 = np.array([1.3, 2.7, 3.8, 4.7,  4.])

        self.ex_PSD = np.array([[1.1,  1.2, 1.2, 1.3],
                               [2.4,  2.7, 2.4, 2.7],
                               [3.95, 3.9, 4,   3.8],
                               [5,    5.4, 5.7, 4.7],
                               [5,    6,   6,   4]])
        # Create the HDF5 file test1
        with h5py.File(self.testv1, 'w') as fh:
            fh.create_group('PSD')
            fh.create_dataset('f', data=self.ex_f)
            fh.create_dataset('PSD/d001', data=self.d1)
            fh.create_dataset('PSD/d002', data=self.d2)
            fh.create_dataset('PSD/d003', data=self.d3)
            fh.create_dataset('PSD/d004', data=self.d4)

        # Create the test HDF5 file test2
        with h5py.File(self.testv2, 'w') as fh:
            fh['f'] = self.ex_f
            fh['PSD'] = self.ex_PSD

    def tearDown(self):
        silentremove(self.testv1)
        silentremove(self.testv2)
        silentremove(self.converted)

    # def test_old_data_format_error(self):
    #     """Make sure that the function get_data throws a useful error when it
    #     encounters an file with a PSD group."""
    #     assert_raises(ValueError, get_data, self.testv1)

    # def test_convert_data(self):
    #     convert_data(self.testv1, self.converted)
    #     f, psd_mean, psd_ci = get_data(self.converted)

    #     ex_mean = np.array([1.2, 2.55, 3.9125, 5.2, 5.25])
    #     ex_ci = np.array([0.080016664930917122, 0.16974097914175013,
    #                   0.083683431255336824, 0.43090292797024871,
    #                   0.93827856560121137])

    #     assert np.allclose(self.ex_f, f)
    #     assert np.allclose(ex_mean, psd_mean)
    #     assert np.allclose(ex_ci, psd_ci)

    # # Test should be skipped for now
    # def test_covert_data_new_format(self):
    #     assert_raises(ValueError, convert_data, self.testv2, self.converted)


def test_fit_residuals():
    sorted_normal = np.load('normal-residuals.npy')  # Load normal residuals
    loc, scale = fit_residuals(sorted_normal)
    assert_almost_equal(loc, 0, places=2)
    assert_almost_equal(scale, 1, places=2)
