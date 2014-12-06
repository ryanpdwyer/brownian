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
import click
import os
import docutils.core
import os.path
import io

import bs4
import base64

def img2uri(html_text):
    """Takes text html formatted input, and converts any img tags to inline
    data uri. Return html_text which can be written to a single, self-contained
    file"""

    soup = bs4.BeautifulSoup(html_text)

    image_tags = soup.find_all('img')

    for image_tag in image_tags:
        image_path = image_tag.attrs['src']
        if 'http' not in image_path:
            base64_text = base64.b64encode(open(image_path, 'rb').read())
            ext = os.path.splitext(image_path)[1][1:]
            
            image_tag.attrs['src'] = (
                "data:image/{ext};base64,{base64_text}".format(
                    ext=ext, base64_text=base64_text)
                )

    return soup.prettify("utf-8")

from brownian import u, Cantilever, BrownianMotionFitter, get_data

# Cross-platform configuration file location
config_file = os.path.expanduser("~/foo.ini")

def calck(filename, f_min, f_max,
        quality_factor, spring_constant, resonance_frequency, temperature):
    est_cant = Cantilever(f_c=resonance_frequency*u.Hz,
                          k_c=spring_constant*u.N/u.m,
                          Q=quality_factor*u.dimensionless)

    data = get_data(filename)
    bmf = BrownianMotionFitter(*data, T=temperature, est_cant=est_cant)
    bmf.calc_fit(f_min, f_max)
    return bmf


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.argument('f_min', type=float)
@click.argument('f_max', type=float)
@click.option('--quality-factor', '-Q', default=30000.0, help="Est. quality factor [unitless]")
@click.option('--spring-constant', '-k', default=3.5, help="Est. spring const. [N/m]")
@click.option('--temperature', '-T', default=298, help='Temperature (298) [K]')
@click.option('--resonance-frequency', '-f', default=None, type=float, help="Est. cant. freq. [Hz]")
def cli(filename, f_min, f_max,
        quality_factor, spring_constant, resonance_frequency, temperature):
    if resonance_frequency is None:
        resonance_frequency = (f_high + f_low) / 2.0


    file_path = os.path.abspath(filename)
    cli_reproduce = """calck "{file_path}" {f_min} {f_max} -Q {Q} -k {k} -f {f} -T {T}""".format(file_path=file_path, f_min=f_min, f_max=f_max, Q=quality_factor,
    k=spring_constant, f=resonance_frequency, T=temperature) 
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

    fit_file = ".fit.png"
    reduced_residuals_file = ".reduced_residuals.png"
    cdf_file = ".cdf.png"

    bmf.plot_fit(fit_file)
    bmf.plot_reduced_residuals(reduced_residuals_file)
    bmf.plot_cdf(cdf_file)

    ReST = u"""\
Report
======


{report}

.. image:: {fit_file}

.. image:: {reduced_residuals_file}

.. image:: {cdf_file}

Reproduce this analysis with the following shell command:

.. code:: bash
    
    {cli_reproduce}

or with the following python code:

.. code:: python

{python_reproduce} 


""".format(fit_file=fit_file, reduced_residuals_file=reduced_residuals_file,
    cdf_file=cdf_file, cli_reproduce=cli_reproduce,
    python_reproduce=python_reproduce, report = bmf.rst_report())
    
    # io module instead of built-in open because it allows specifying encoding
    # See http://stackoverflow.com/a/22288895/2823213
    with io.open('report.rst', 'w', encoding='utf8') as f:
        f.write(ReST)
    
    image_dependent_html = docutils.core.publish_string(ReST, writer_name='html')
    self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')
    
    with io.open('report.html', 'w', encoding='utf8') as f:
        f.write(self_contained_html)



    click.echo(cli_reproduce)
    click.echo(python_reproduce)
    click.echo(bmf.report())