import h5py
import numpy as np
from brownian import silentremove
from nose.tools import assert_almost_equal, ok_
from numpy.testing import assert_array_almost_equal
import unittest
from brownian import hdf5
from brownian.hdf5 import generate_sample_file
import os


class Test_generate_sample_file(unittest.TestCase):
    """Test constructing a sample hdf5 file, to verify the file
    is constructed correctly."""
    filename = 'test-pos-PSD-1.h5'
    x = np.array([0, 0.5, 1.0, 1.5])
    y = np.array([1, 2.1, 2.9, 4])
    y_std = np.array([0.2, 0.3, 0.4, 0.5])

    file_attrs = {'date': "2014-07-23",
                  'time': '13:00:00',
                  'version': 'pos-PSD-1.0',
                  'instrument': 'PSB-B19-AFM',
                  'help': '''Nice help about the data structure; \
omitted here for brevity.'''}

    x_attrs = {'name': 'Frequency',
               'unit': 'Hz',
               'label': 'Frequency [Hz]',
               'label_latex': r'$f \: [\mathrm{Hz}]',
               'help': 'Frequency array for PSD.'}

    y_attrs = {'name': 'PSD',
               'n_avg': 4,
               'units': 'nm^2/Hz',
               'label': 'PSD [nm^2/Hz]',
               'label_latex': r'P_{\delta x} \: [\mathrm{nm}^2/\mathrm{Hz}]',
               'help': 'Power spectral density of position fluctuations.'}

    def setUp(self):
        generate_sample_file(self.filename)
        self.fh = h5py.File(self.filename, 'r')

    def test_datasets_groups(self):
        fh = self.fh
        ok_(isinstance(fh['x'], h5py.Dataset), "x should be a dataset")
        ok_(isinstance(fh['y'], h5py.Dataset), "f should be a dataset")

        assert_array_almost_equal(fh['x'].value, self.x)
        assert_array_almost_equal(fh['y'].value, self.y)
        assert_array_almost_equal(fh['y_std'].value, self.y_std)

    def test_attributes(self):
        fh = self.fh
        assert fh.attrs.items() == self.file_attrs.items()
        assert fh['x'].attrs.items() == self.x_attrs.items()
        assert fh['y'].attrs.items() == self.y_attrs.items()

    def tearDown(self):
        self.fh.close()
        silentremove(self.filename)
