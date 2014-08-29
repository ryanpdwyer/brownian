import h5py
import numpy as np
from brownian import silentremove
from nose.tools import assert_almost_equal, ok_
from numpy.testing import assert_array_almost_equal
import unittest
from brownian.hdf5 import generate_sample_file
import os


class Test_generate_sample_file(unittest.TestCase):
    """Test constructing a sample hdf5 file, to verify the file
    is constructed correctly."""
    filename = 'test-pos-PSD-1.h5'
    f = np.array([0, 0.5, 1.0, 1.5])
    PSD_mean = np.array([1, 2.1, 2.9, 4])
    PSD_stdev = np.array([0.2, 0.3, 0.4, 0.5])

    file_attrs = {'date': "2014-07-23",
                  'time': '13:00:00',
                  'version': 'pos-PSD-1.0',
                  'instrument': 'PSB-B19-AFM',
                  'help': '''Nice help about the data structure; \
omitted here for brevity.'''}
    f_attrs = {'unit': 'Hz',
               'help': 'Frequency array for PSD.'}

    PSD_attrs = {'n': 4,
                 'units': 'nm^2/Hz',
                 'help': 'Power spectral density of position fluctuations.'}

    def setUp(self):
        generate_sample_file(self.filename)
        self.fh = h5py.File(self.filename, 'r')

    def test_datasets_groups(self):
        fh = self.fh
        ok_(isinstance(fh['PSD'], h5py.Group), "PSD should be a group")
        ok_(isinstance(fh['f'], h5py.Dataset), "f should be a dataset")

        assert_array_almost_equal(fh['f'].value, self.f)
        assert_array_almost_equal(fh['PSD/mean'].value, self.PSD_mean)
        assert_array_almost_equal(fh['PSD/stdev'].value, self.PSD_stdev)

    def test_attributes(self):
        fh = self.fh
        assert fh.attrs.items() == self.file_attrs.items()
        assert fh['f'].attrs.items() == self.f_attrs.items()
        assert fh['PSD'].attrs.items() == self.PSD_attrs.items()

    def tearDown(self):
        silentremove(self.filename)
