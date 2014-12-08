# -*- coding: utf-8 -*-
"""
calck
=====

This is a command line tool to easily fit brownian motion data and output
a report. The default usage of the command should be,

.. code: bash

    calck file.h5 66300 66500 -Q 30000 -k 3.5

Should defaults be saved into a .calckrc file?

Yes, I think so. Then, Q, k, f_c can be filled in from the config file
if necessary.
"""
import os
import os.path
import io
import zipfile
import base64

from bunch import Bunch
import click
import h5py
import docutils.core

import bs4

import brownian
from brownian import u, Cantilever, BrownianMotionFitter, get_data, silentremove

def file_extension(filename):
    """Return the file extension for a given filename. For example,

    Input               Output
    -------------       ---------
    data.csv            csv
    data.tar.gz         gz
    .vimrc              (empty string)"""

    return os.path.splitext(filename)[1][1:]

def img2uri(html_text):
    """Convert any relative reference img tags in html_input to inline data uri.
    Return the transformed html, in utf-8 format."""

    soup = bs4.BeautifulSoup(html_text)

    image_tags = soup.find_all('img')

    for image_tag in image_tags:
        image_path = image_tag.attrs['src']
        if 'http' not in image_path:
            base64_text = base64.b64encode(open(image_path, 'rb').read())
            ext = file_extension(image_path)
            
            image_tag.attrs['src'] = (
                "data:image/{ext};base64,{base64_text}".format(
                    ext=ext, base64_text=base64_text)
                )

    return soup.prettify("utf-8")

# Cross-platform configuration file location
config_file = os.path.expanduser("~/foo.ini")

def calck(filename, f_min, f_max,
        quality_factor, spring_constant, resonance_frequency, temperature):
    """Base function for the command line interface. TODO: Add numpy style
    param strings."""
    est_cant = Cantilever(f_c=resonance_frequency*u.Hz,
                          k_c=spring_constant*u.N/u.m,
                          Q=quality_factor*u.dimensionless)

    data = get_data(filename)
    bmf = BrownianMotionFitter(*data, T=temperature, est_cant=est_cant)
    bmf.calc_fit(f_min, f_max)
    return bmf

"""CLI object psuedo-code:
class BrownianCLI(object):
    def __init__(self, filename, f_min, f_max, output,
        quality_factor, spring_constant, resonance_frequency, temperature):
        pass
        # boilerplate, plus, pick resonance frequency
    

    def __repr__(self):
        return cli_reproduce stuff

    def report(self):
        if output is None:
            self._text_report()
        elif output == 'html'
            self._html_report()
        elif output == 'zip'
            self._zip_report()

    def _text_report(self):
        pass

    def _html_report(self):
        pass

    def _zip_report(self):
        pass
"""

@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.argument('f_min', type=float)
@click.argument('f_max', type=float)
@click.option('--output', '-o', type=str, default=None)
@click.option('--quality-factor', '-Q', default=30000.0, help="Est. quality factor [unitless]")
@click.option('--spring-constant', '-k', default=3.5, help="Est. spring const. [N/m]")
@click.option('--temperature', '-T', default=298, help='Temperature (298) [K]')
@click.option('--resonance-frequency', '-f', default=None, type=float, help="Est. cant. freq. [Hz]")
def cli(filename, f_min, f_max, output,
        quality_factor, spring_constant, resonance_frequency, temperature):
    # This function needs to be broken up into component pieces.
    # For example, the reproduce python, reproduce cli bits can easily be their
    # own functions (and in fact, may make more sense as __repr__ methods of
    # classes)
    # The report generator should also be its own function.
    if resonance_frequency is None:
        resonance_frequency = (f_high + f_low) / 2.0


    file_path = os.path.abspath(filename)
    with h5py.File(filename, 'r') as f:
        date = f.attrs['date']
        time = f.attrs['time'].replace(':', '')

    if output is None:
        cli_reproduce = """calck "{file_path}" {f_min} {f_max} -Q {Q} -k {k} -f {f} -T {T}""".format(file_path=file_path, f_min=f_min, f_max=f_max, Q=quality_factor,
    k=spring_constant, f=resonance_frequency, T=temperature)
    else:
        cli_reproduce = """calck "{file_path}" {f_min} {f_max} -o {output} -Q {Q} -k {k} -f {f} -T {T}""".format(file_path=file_path, f_min=f_min, f_max=f_max, Q=quality_factor,
    k=spring_constant, f=resonance_frequency, T=temperature, output=output) 
    
    python_reproduce = """\
    import brownian._calck
    bmf = brownian._calck.calck("{file_path}",
        f_min={f_min}, f_max={f_max},
        quality_factor={Q},
        spring_constant={k},
        resonance_frequency={f},
        temperature={T})

    print(bmf.report())
""".format(file_path=file_path, f_min=f_min, f_max=f_max, Q=quality_factor,
    k=spring_constant, f=resonance_frequency, T=temperature)    

    bmf = calck(filename, f_min, f_max,
        quality_factor, spring_constant, resonance_frequency, temperature)


    if output is None:
        text = u"""\
Generated by brownian version {version}.

Reproduce this analysis with the following shell command:
    
{cli_reproduce}

or with the following python code:

{python_reproduce}

{report}
""".format(cli_reproduce=cli_reproduce,
    python_reproduce=python_reproduce, report = bmf.rst_report(),
    version=brownian.__version__)
    
        click.echo(text)
        return 0
    else:

        files = Bunch(fit="fit.png", reduced_residuals="reduced_residuals.png",
            cdf='cdf.png')

        bmf.plot_fit(files.fit)
        bmf.plot_reduced_residuals(files.reduced_residuals)
        bmf.plot_cdf(files.cdf)

        ReST = u"""\
Report
======


{report}

.. image:: {files.fit}

.. image:: {files.reduced_residuals}

.. image:: {files.fit}

Reproduce this analysis with the following shell command:

.. code:: bash
    
    {cli_reproduce}

or with the following python code:

.. code:: python

{python_reproduce} 


Generated by ``brownian`` version ``{version}``.

""".format(files=files, cli_reproduce=cli_reproduce,
    python_reproduce=python_reproduce, report = bmf.rst_report(),
    version=brownian.__version__)
        
        ext = file_extension(output)
        if ext not in ('zip', 'html'):
            click.echo('output file extension must be zip or html, not {0}'.format(ext))
            return -1
        elif ext == 'html':
            image_dependent_html = docutils.core.publish_string(ReST, writer_name='html')
            self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')
    
            # io module instead of built-in open because it allows specifying
            # encoding. See http://stackoverflow.com/a/22288895/2823213
            with io.open(output, 'w', encoding='utf8') as f:
                f.write(self_contained_html)

            for filename in files.values():
                silentremove(filename)
            return 0

        elif ext == 'zip':
            files.html = date+'-'+time+'.html'
            files.rst = date+'-'+time+'.rst'

            image_dependent_html = unicode(
                docutils.core.publish_string(ReST, writer_name='html'), 'utf8'
                )
    
            with io.open(files.html, 'w', encoding='utf8') as f:
                f.write(image_dependent_html)

            with io.open(files.rst, 'w', encoding='utf8') as f:
                f.write(ReST)

            
            with zipfile.ZipFile(output, 'w') as z:
                for f in files.values():
                    z.write(f)

            # Delete non-zipped version of the files
            for f in files.values():
                silentremove(f)

            return 0
