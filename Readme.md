Brownian
========

[![Build Status](https://travis-ci.org/ryanpdwyer/brownian.svg?branch=master)](https://travis-ci.org/ryanpdwyer/brownian)
[![Documentation Status](https://readthedocs.org/projects/brownian/badge/?version=latest)](https://readthedocs.org/projects/brownian/?badge=latest)

The package `brownian` fits scanned probe microscopy cantilever Brownian motion data, which allows us to calculate the resonance frequency, spring constant and quality factor of the cantilever, along with the noise floor of the detector.

Windows installation
--------------------

To install on Windows, follow the instructions at StackOverflow for installing Theano (http://stackoverflow.com/a/33706634), then do,

```
pip install git+https://github.com/pymc-devs/pymc3
```

Then you should be able to install by cloning the respository and running `python setup.py install`, or `pip install git+https://github.com/ryanpdwyer/brownian`.

Features
--------

- Read data from a properly structured HDF5 file
- Fit data to determine cantilever resonance frequence $f_c$, spring constant $k_c$, and qualify factor $Q$.
- Output the results of the fit to a nicely labeled HDF5 group (maybe?)
- Output the results of the fit to a markdown or text file.
