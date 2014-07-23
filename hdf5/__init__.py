"""
brownian.hdf5
2014-07-23
Ryan Dwyer

This contains code to implement the hdf5 file structure used for the data
we work up. We will generate the example file using h5py, and then avoid having
to version control the binary hdf5.
"""

import h5py
import numpy as np

def generate_sample_file(filename):
    fh = h5py.File(filename, 'w')
    fh.close()