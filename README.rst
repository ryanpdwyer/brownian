Brownian
========

.. image:: https://travis-ci.org/ryanpdwyer/brownian.svg?branch=master
    :target: https://travis-ci.org/ryanpdwyer/brownian

The package ``brownian`` fits scanned probe microscopy cantilever Brownian motion data, which allows us to calculate the resonance frequency, spring constant and quality factor of the cantilever, along with the noise floor of the detector.

Windows installation
--------------------

To install on Windows, follow the instructions at StackOverflow for installing Theano (http://stackoverflow.com/a/33706634), then do,

.. code::

    pip install git+https://github.com/pymc-devs/pymc3

Then you should be able to install by cloning the respository and running ``python setup.py install``, or ``pip install git+https://github.com/ryanpdwyer/brownian``.

Features
--------

- Fit data to determine cantilever resonance frequency, spring constant, and qualify factor.
- Output the results of the fit to an html report.
