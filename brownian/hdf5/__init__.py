"""
brownian.hdf5
2014-07-23
Ryan Dwyer

This contains code to implement the hdf5 file structure used for the data
we work up. We will generate the example file using h5py, and then avoid having
to version control the binary hdf5.

HDF5 Format for Brownian Motion Data
====================================

Here is a sample of the HDF5 file format.

```
/
attributes: {'date': '2014-07-22',
             'time': '13:00:00',
             'version': 'pos-PSD-1.0',
             'instrument': 'PSB-B19-AFM',
             'help': 'Nice help about the data structure; omitted here for brevity.'}
        f
        attributes: {'unit': 'Hz',
                     'help': 'Frequency array for PSD.'}
        [0, 0.5, 1, 1.5]

    PSD/
    attributes: {'n': 4,
                 'units': 'nm^2/Hz',
                 'help': 'Power spectral density of position fluctuations.'}
        mean
            [1, 2.1, 2.9, 4]
        stdev
            [0.2, 0.3, 0.4, 0.5]

```
"""

import h5py
import numpy as np


def generate_sample_file(filename):
    fh = h5py.File(filename, 'w')

    # Set file attributes
    file_attrs = {'date':"2014-07-23",
                  'time': '13:00:00',
                  'version': 'pos-PSD-1.0',
                  'instrument': 'PSB-B19-AFM',
                  'help': '''Nice help about the data structure; \
omitted here for brevity.'''}
    update_h5py_attrs(fh.attrs, file_attrs)

    # Set frequency information
    fh['x'] = np.array([0, 0.5, 1.0, 1.5])

    x_attrs = {'name': 'Frequency',
               'unit': 'Hz',
               'label': 'Frequency [Hz]',
               'label_latex': r'$f \: [\mathrm{Hz}]',
               'help': 'Frequency array for PSD.'}
    update_h5py_attrs(fh['x'].attrs, x_attrs)

    # Set PSD information
    fh['y'] = np.array([1, 2.1, 2.9, 4])
    y_attrs = {'name': 'PSD',
               'n_avg': 4,
               'units': 'nm^2/Hz',
               'label': 'PSD [nm^2/Hz]',
               'label_latex': r'P_{\delta x} \: [\mathrm{nm}^2/\mathrm{Hz}]',
               'help': 'Power spectral density of position fluctuations.'}

    update_h5py_attrs(fh['y'].attrs, y_attrs)

    fh['y_std'] = np.array([0.2, 0.3, 0.4, 0.5])

    fh.close()


def update_h5py_attrs(h5py_attrs, attrs):
    """Update the h5py attributes in h5py_attrs by adding the attributes in 
    the dictionary attrs.

    This will overwrite existing attributes.
    """
    for key, val in attrs.viewitems():
        h5py_attrs[key] = val
