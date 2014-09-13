#!/usr/bin/env python
# Enables the package to be used in develop mode
# See http://pythonhosted.org/setuptools/setuptools.html#development-mode
# See https://github.com/scikit-learn/scikit-learn/issues/1016
try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup

setup(name='brownian',
      description="",
      version="0.1dev",
      author='Ryan Dwyer',
      author_email='ryanpdwyer@gmail.com',
      packages=['brownian'],
      install_requires=[
      'numpy', 'scipy', 'matplotlib', 'pint',
      'nose', 'cython', 'h5py', 'uncertainties'
      ])
