Brownian
========

[![Build Status](https://travis-ci.org/ryanpdwyer/brownian.svg?branch=master)](https://travis-ci.org/ryanpdwyer/brownian)

The package `brownian` fits scanned probe microscopy cantilever Brownian motion data, which allows us to calculate the resonance frequency, spring constant and quality factor of the cantilever, along with the noise floor of the detector.

Features
--------

- Read data from a properly structured HDF5 file
- Fit data to determine cantilever resonance frequence $f_c$, spring constant $k_c$, and qualify factor $Q$.
- Output the results of the fit to a nicely labeled HDF5 group (maybe?)
- Output the results of the fit to a markdown or text file.
