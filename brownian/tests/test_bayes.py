import unittest
import time
from nose.tools import assert_raises, assert_almost_equal
from numpy.testing import assert_array_almost_equal
import h5py

from brownian import bayes
from brownian import silentremove



class testACmdStan(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()
        self.fh = h5py.File('../ex/brownian173033.h5', 'r')

    def test_integration_cmd_stan(self):
        f = self.fh['x'][:]
        PSD = self.fh['y'][:]
        m = (f > 69500) & (f < 71500)
        d = bayes.np2data(f, PSD, 32, fmin=70400, fmax=70700,
              kc=3.5, Q=20000, sigma_Q=15000, sigma_kc=2.5,
              Pdet=1e-8, sigma_Pdet=3e-8, # Pdet, sigma_Pdet not required
             )
        df_samples = bayes.cmdstan_sample(d, 10, 2,)
        ppb = bayes.PlotCmdStanBrownian(d, df_samples, 'test-cmdstan')

    def tearDown(self):
        self.fh.close()
        silentremove('gamma-1.csv')
        silentremove('gamma-2.csv')
        silentremove('data.dump')
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

class testBayes(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()
        self.fh = h5py.File('../ex/brownian173033.h5', 'r')
    def test_integration(self):
        """Integration test: data extraction, sampling, and creation of plotting object."""
        d = bayes.fh2data(self.fh, 70521, 70601, 3.5, 20000,
              sigma_Q=15000, Pdet=1e-8,
              sigma_Pdet=3e-8, sigma_kc=2.5)
        traces = bayes.sample_pymc3(d, samples=10, njobs=1)
        ppb = bayes.PlotPyMCBrownian(d, traces, 'test-data')

    def tearDown(self):
        self.fh.close()
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))




