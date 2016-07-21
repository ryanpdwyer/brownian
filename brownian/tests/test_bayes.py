import unittest
from nose.tools import assert_raises, assert_almost_equal
from numpy.testing import assert_array_almost_equal
import h5py

from brownian import bayes

class testBayes(unittest.TestCase):
    def setUp(self):
        self.fh = h5py.File('../ex/brownian173033.h5', 'r')
    def test_integration(self):
        """Integration test: data extraction, sampling, and creation of plotting object."""
        d = bayes.fh2data(self.fh, 70511, 70611, 3.5, 20000,
              sigma_Q=15000, Pdet=1e-8,
              sigma_Pdet=3e-8, sigma_kc=2.5)
        traces = bayes.sample_pymc3(d, samples=10, njobs=2)
        ppb = bayes.PlotPyMCBrownian(d, traces, 'test-data')

    def tearDown(self):
        self.fh.close()
