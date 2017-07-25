#!/usr/bin/env python
# Enables the package to be used in develop mode
# See http://pythonhosted.org/setuptools/setuptools.html#development-mode
# See https://github.com/scikit-learn/scikit-learn/issues/1016
import os
import platform
from setuptools import setup

uname = platform.uname()
system_name = uname[0]
if system_name == "Windows":
    script_extension = ".exe"
elif system_name == "Darwin":
    script_extension = ".osx"
elif system_name == "Linux":
    script_extension = ".linux"

# See https://github.com/warner/python-versioneer
import imp
fp, pathname, description = imp.find_module('versioneer')
try:
    versioneer = imp.load_module('versioneer', fp, pathname, description)
finally:
    if fp: fp.close()


versioneer.VCS = 'git'
versioneer.versionfile_source = 'brownian/_version.py'
versioneer.versionfile_build = 'brownian/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'brownian-' # dirname like 'myproject-1.2.0'

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
      version=versioneer.get_version(),
      author='Ryan Dwyer',
      author_email='ryanpdwyer@gmail.com',
      packages=get_packages('brownian'),
      install_requires=[
      'numpy', 'scipy', 'matplotlib', 'pint',
      'nose', 'h5py', 'uncertainties', 'click', 'BeautifulSoup4', 'docutils',
      'bunch', 'six', 'psutil', 'kpfm'
      ],
      package_data={'brownian': ['stanmodels/gamma{}'.format(script_extension),
                                  'stanmodels/gamma2{}'.format(script_extension),
                                  'ex/brownian-red.h5']},
      tests_require=['nose'],
      test_suite='brownian.tests.discover',
      license = 'MIT',
      entry_points="""
        [console_scripts]
        calck=brownian._calck:cli
        bayesk=brownian.bayes:pymc_brownian_cli
        stan_calck=brownian.bayes:cmdstan_brownian_cli
      """,
      cmdclass=versioneer.get_cmdclass(),
      )
