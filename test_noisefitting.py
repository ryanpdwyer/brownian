from noisefitting import BrownianMotionFitter, u, calc_k_c
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
from nose.tools import assert_raises, assert_almost_equal
import unittest
from uncertainties import ufloat

os.chdir(os.path.dirname(os.path.realpath(__file__)))
data = pd.read_pickle('data.pkl')
f = data.f.values
PSDx = data.PSDx.values
estimates = {'f_c': 63700, 'k_c': 3.5, 'Q': 20000}
T = 295


def test_BrownMotionFitter_init():
    # Try a case that should work.
    reduced_f = f[::25000]
    reduced_PSD = PSDx[::25000]
    a = BrownianMotionFitter(reduced_f, reduced_PSD, T, estimates)
    assert np.all(a.f == reduced_f)
    assert np.all(a.PSD_raw == reduced_PSD)
    assert np.all(a.T == T * u.K)
    assert a.estimates == estimates
    # Try a case that should fail.
    bad_estimates = {'f_c': 63700, 'k_c': 3.5}
    assert_raises(ValueError, BrownianMotionFitter,
                            reduced_f, reduced_PSD, T, bad_estimates)


ex_scaled_PSD = np.load('scale_data_PSD.npy')

class testBrownianMotionFitter_fitting(unittest.TestCase):
    def setUp(self):
        self.bmf = BrownianMotionFitter(f, PSDx, T, estimates)

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
    P_x0 = ufloat(1.75789868673e-12, 1.75789868673e-14) * u.nm ** 2 / u.Hz # 1/100
    f_c = ufloat(50000, 0.5) * u.Hz # 1/100000 relative
    Q = ufloat(10000, 100) * u.dimensionless # 1/100
    T = ufloat(300, 3) * u.K  # 1/100
    ex_k_c = ufloat(3, 0.05196153288731964) * u.N/u.m
    k_c = calc_k_c(f_c, Q, P_x0, T)
    assert_almost_equal(k_c.magnitude.n, ex_k_c.magnitude.n)
    assert_almost_equal(k_c.magnitude.s, ex_k_c.magnitude.s)


