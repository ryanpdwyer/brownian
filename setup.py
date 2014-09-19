#!/usr/bin/env python
# Enables the package to be used in develop mode
# See http://pythonhosted.org/setuptools/setuptools.html#development-mode
# See https://github.com/scikit-learn/scikit-learn/issues/1016
try:
    import setuptools
except ImportError:
    pass

from distutils.core import setup

def get_packages(package):
    """
    Return root package and all sub-packages.
    See https://github.com/un33k/django-uuslug/blob/master/setup.py.
    Using this to auotmatically find all packages.

    """
    return [dirpath
            for dirpath, dirnames, filenames in os.walk(package)
            if os.path.exists(os.path.join(dirpath, '__init__.py'))]

setup(name='brownian',
      description="",
      url='https://github.com/ryanpdwyer/brownian',
      version="0.1dev",
      author='Ryan Dwyer',
      author_email='ryanpdwyer@gmail.com',
      packages=get_packages('brownian'),
      install_requires=[
      'numpy', 'scipy', 'matplotlib', 'pint',
      'nose', 'h5py', 'uncertainties'
      ])
