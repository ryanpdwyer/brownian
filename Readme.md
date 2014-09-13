Brownian
========

This package should implement John Marohn's Brownian motion fitting, which allows us to calculate the spring constant of a cantilever, in python.

Features
--------

- Read data from a properly structured HDF5 file
- Fit data to determine cantilever resonance frequence $f_c$, spring constant $k_c$, and qualify factor $Q$.
- Output the results of the fit to a nicely labeled HDF5 group (maybe?)
- Output the results of the fit to a markdown or text file.
