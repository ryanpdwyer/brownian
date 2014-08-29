#!/usr/bin/env python
#
# :ref: http://docs.fabfile.org/en/1.4.1/tutorial.html
# :ref: http://ipython.org/ipython-doc/1/interactive/nbconvert.html

"""
This is a fabric file to help compile sphinx HTML documentation for the
project.  The following commands are defined::

    fab cleab
    fab html
    fab show

"""

from fabric.api import *
# from fabric.context_managers import lcd
# import glob
import os
import webbrowser

# env.ipython_dir = os.path.join(os.path.join('..','freqdemod'),'docs')
# home = os.getcwd()


def help():
    """Print out a helpful message."""

    print("=====================================================================")
    print("fab clean      Delete the contents of the _build/ directory")
    print("fab html       Create sphinx documentation as stand-alone HTML files")
    print("fab show       Open the HTML documentation in a web browser.")
    print("====================================================================")

def clean():
    """Delete the contents of the _build/ directory."""

    local('rm -rf _build/*')

# def html_full():
#     """
#     Convert the ipynb files in ``env.ipython_dir`` to HTML files, then
#     create sphinx documentation as stand-alone HTML files.
#     """


#     os.chdir(os.path.join('{ipython_dir}'.format(**env)))
#     ipynb_files = glob.glob('*.ipynb')
#     os.chdir(home)

#     with lcd('{ipython_dir}'.format(**env)):
#         local('ls -la')
#         for file in ipynb_files:
#             print '{}'.format(file)
#             local('ipython nbconvert --to html --template full {}'.format(file))  # noqa

#     with lcd(''):
#         local('ls -la')
#         local('sphinx-build -b html . _build/html')

#     print "Build finished; see _build/html/index.html"

def html():
    """Create sphinx documentation as stand-alone HTML files."""


    local('ls -la')
    local('sphinx-build -b html . _build/html')

    print "Build finished; see _build/html/index.html"
    
def show():
    """Open the HTML documentation in a cross-platform way."""
    cwd = os.getcwd()
    index_path = "file://{cwd}/_build/html/index.html".format(cwd=cwd)
    webbrowser.open(index_path)
