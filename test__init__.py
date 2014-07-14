from brownian import (BrownianMotionFitter, u, calc_k_c,
                          translate_fit_parameters, average_data, get_data,
                          fit_residuals)
import pandas as pd
import numpy as np
import os
from nose.tools import assert_raises, assert_almost_equal
import unittest
from uncertainties import ufloat
from numpy.testing import assert_array_almost_equal
from jittermodel.ubase import UnitCantilever

os.chdir(os.path.dirname(os.path.realpath(__file__)))
data = pd.read_pickle('data.pkl')
f = data.f.values
PSDx = data.PSDx.values
PSD_wgt = data.PSDx.values * 0.1
est_cant = UnitCantilever(f_c=63700*u.Hz, k_c=3.5*u.N/u.m,
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
        self.bmf.calc_initial_params()

        ex_initial_params = [8.9173850434295625e-05, 63700, 20000]

        np.testing.assert_allclose(ex_initial_params, self.bmf.initial_params)



def test_first_pass():
    pass


def test_Px0():
    pass


def test_Pf():
    pass


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


def test_average_data():
    data = np.array([[1.1, 2.4, 3.95, 5.],
                     [1.2, 2.7, 3.9, 5.4],
                     [1.2, 2.4, 4., 5.7],
                     [1.3, 2.7, 3.8, 4.7]])
    ex_avg = np.array([1.2, 2.55, 3.9125, 5.2])
    ex_ci = np.array([0.08001666, 0.16974098, 0.08368343, 0.43090293])
    avg, ci = average_data(data)
    assert_array_almost_equal(ex_avg, avg)
    assert_array_almost_equal(ex_ci, ci)


def test_get_data():
    ex_f = np.array([1, 2, 3, 4, 5])
    # ex_PSD = np.array([[1.1, 2.4, 3.95, 5., 5],
    #                   [1.2, 2.7, 3.9, 5.4, 6],
    #                   [1.2, 2.4, 4., 5.7, 6],
    #                   [1.3, 2.7, 3.8, 4.7, 4]])
    ex_mean = np.array([1.2, 2.55, 3.9125, 5.2, 5.25])
    ex_ci = np.array([0.080016664930917122, 0.16974097914175013,
                      0.083683431255336824, 0.43090292797024871,
                      0.93827856560121137])
    f, psd_mean, psd_ci = get_data('test.h5')
    assert np.allclose(ex_f, f, 1e-4)
    assert np.allclose(ex_mean, psd_mean)
    assert np.allclose(ex_ci, psd_ci)


def test_fit_residuals():
    sorted_normal = np.load('normal-residuals.npy')  # Load normal residuals
    loc, scale = fit_residuals(sorted_normal)
    assert_almost_equal(loc, 0, places=2)
    assert_almost_equal(scale, 1, places=2)
