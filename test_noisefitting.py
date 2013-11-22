from noisefitting import BrownianMotionFitter, u
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
from nose.tools import assert_raises

os.chdir(r'C:\Users\as-chm-marohninst\Dropbox\Python Code\projects\NoiseFitting')
data = pd.read_pickle('data.pkl')
f = data.f.values
PSDx = data.PSDx.values
estimates = {'f_c':63700, 'k_c':3.5, 'Q':20000}
T = 295

def test_NoiseFitter_init():
    # Try a case that should work.
    reduced_f = f[::25000]
    reduced_PSD = PSDx[::25000]
    a = BrownianMotionFitter(reduced_f, reduced_PSD, T, estimates)
    assert np.all(a.f == reduced_f)
    assert np.all(a.PSD_raw == reduced_PSD)
    assert np.all(a.T == T * u.K) 
    assert a.estimates == estimates
    # Try a case that should fail.
    bad_estimates = {'f_c':63700, 'k_c':3.5}
    assert_raises(ValueError, BrownianMotionFitter,
                            reduced_f, reduced_PSD, T, bad_estimates)
    

def test_Px0():
	pass

def test_Pf():
	pass