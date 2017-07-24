# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import os
import io

import datetime
import copy
import platform
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import docutils.core
import numpy as np
import click
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import h5py
import pymc3 as pm
import pymc3.stats as pmstats
from brownian import u, calc_P_x0, Pf
from brownian._rdump import dump_to_rdata
sns.set_style("white")
from matplotlib.offsetbox import AnchoredText
from six.moves import cPickle as pickle
from statsmodels.nonparametric.smoothers_lowess import lowess
from six import string_types
from brownian._calck import img2uri
import subprocess
import psutil
from cStringIO import StringIO
import pandas as pd
from kpfm.util import txt_filename

# print("Platform information:\n\n{}".format(platform.platform()))
uname = platform.uname()
system_name = uname[0]
if system_name == "Windows":
    script_extension = "exe"
elif system_name == "Darwin":
    script_extension = "osx"
elif system_name == "Linux":
    script_extension = "linux"
else:
    raise UserWarning("Cannot use cmdstan compiled programs on this system.")

windows = 'Windows' == platform.system()

directory = os.path.split(__file__)[0]
model_code_dict_fname = os.path.join(directory, 'stan_model_code.pkl')

def gamma_mustd(mu, std):
    """Define the Gamma distribution in terms of its mean ``mu`` and standard deviation ``std``,
    rather than shape ``k``, and scale ``theta`.
    
    See https://en.wikipedia.org/wiki/Gamma_distribution"""
    k = (mu/std)**2
    theta = std**2/mu
    return stats.gamma(a=k, scale=theta)

def Pfi(d, samp, i):
    fc = d['mu_fc'] + samp['dfc'][i]
    Q = samp['Q'][i]
    kc = samp['kc'][i]
    return Pf(d['f'],
              calc_P_x0(fc*u.Hz, Q, kc*u('N/m'), d['T']*u.K).to('nm^2/Hz').magnitude,
              fc, Q, samp['Pdet'][i]*d['scale']
             )

def Pf_func_gamma(x, samp, d):
    """The equation to calculate :math:`P_x(f)` from the cantilever
    parameters and detector noise floor.

    f
        Frequency, the independent variable.

    P_x0
        The zero frequency power spectral density of position fluctuations.

    f_c
        The cantilever resonance frequency.

    Q
        The quality factor of the cantilever.

    P_detector
        The detector noise floor.
        """
    fc = d['mu_fc'] + samp['dfc']
    Q = samp['Q']
    kc = samp['kc']
    return Pf(x,
              calc_P_x0(fc*u.Hz, Q, kc * u('N/m'), d['T']*u.K).to('nm^2/Hz').magnitude,
              fc, Q, samp['Pdet'] * d['scale']
             )


def eval_samples(func, x, samples, data=None, **kwargs):
    """Evaluate the function ``func`` at x, for each sample."""
    N = len(samples.values()[0])
    M = len(x)
    z = np.zeros((N, M))
    for i in xrange(N):
        d = {key: val[i] for key, val in samples.items()}
        z[i, :] = func(x, d, data, **kwargs)
    return z


def fh2data(fh, fmin, fmax, kc, Q, Pdet=None, T=298,
            sigma_fc=5, sigma_kc=5, sigma_Q=10000, sigma_Pdet=None):
    f_all = fh['x'][:]
    m = (f_all > fmin) & (f_all < fmax)
    f = f_all[m]
    psd = fh['y'][:][m]
    M = int(round(fh['y'].attrs['n_avg']))

    N = f.size
    
    fc = f[np.argmax(psd)]

    # Scale data
    psd_scale = psd.mean()
    psd_scaled = psd / psd_scale
    
    if Pdet is None:
        mu_Pdet = np.percentile(psd_scaled, 25)
    else:
        mu_Pdet = Pdet / psd_scale


    if sigma_Pdet is None:
        sigma_Pdet = 5 * mu_Pdet
    else:
        sigma_Pdet = sigma_Pdet / psd_scale
        
    
    return {'fmin': fmin,
     'fmax': fmax,
     'N': N,
     'M': M,       
     'y': psd_scaled,
     'f': f,
     'scale': psd_scale,
     'mu_fc': fc,
     'mu_kc': kc,
     'mu_Q': Q,
     'mu_Pdet': mu_Pdet,  # scaled
     'sigma_fc': sigma_fc,
     'sigma_kc': sigma_kc,
     'sigma_Q': sigma_Q,
     'sigma_Pdet': sigma_Pdet,
     'T': T,
     }


def np2data(freq, PSD, N_averages, fmin, fmax, kc, Q, Pdet=None, T=298,
            sigma_fc=5, sigma_kc=5, sigma_Q=10000, sigma_Pdet=None):
    """
    Create input dictionary for bayesian curve fitting of power spectral density
    brownian motion data. Parameters explained below. Those labeled '(prior)'
    set the prior for the bayesian fitting.

    freq: Array of frequency data [Hz]
    PSD: Mean power spectral density data from N_averages
    N_averages: Number of averages
    fmin: Minimum frequency to fit [Hz]
    fmax: Maximum frequency to fit [Hz]
    kc: Estimated cantilever spring constant [N/m] (prior)
    Q: Estimated cantilever Q (prior)
    Pdet: Estimated detector noise floor. Estimated if not provided. (prior)
    T: Temperature [K] 
    sigma_fc: Standard deviation in estimated cantilever frequency (prior)
    sigma_Q: Standard deviation in estimated quality factor (prior)
    sigma_Pdet: Standard deviation in estimated detector noise [nm^2/Hz]. Estimated if not provided. (prior)
    """
    m = (freq > fmin) & (freq < fmax)
    f = freq[m]
    psd = PSD[m]

    M = int(round(N_averages)) # Force to integer
    N = f.size
    
    fc = f[np.argmax(psd)]

    # Scale data
    psd_scale = psd.mean()
    psd_scaled = psd / psd_scale
    
    if Pdet is None:
        mu_Pdet = np.percentile(psd_scaled, 25)
    else:
        mu_Pdet = Pdet / psd_scale


    if sigma_Pdet is None:
        sigma_Pdet = 5 * mu_Pdet
    else:
        sigma_Pdet = sigma_Pdet / psd_scale
        
    
    return {'fmin': fmin,
     'fmax': fmax,
     'N': N,
     'M': M,       
     'y': psd_scaled,
     'f': f,
     'scale': psd_scale,
     'mu_fc': fc,
     'mu_kc': kc,
     'mu_Q': Q,
     'mu_Pdet': mu_Pdet,  # scaled
     'sigma_fc': sigma_fc,
     'sigma_kc': sigma_kc,
     'sigma_Q': sigma_Q,
     'sigma_Pdet': sigma_Pdet,
     'T': T,
     }



def prior_func_pymc(xname, data, x):
    if xname == 'dfc':
        return norm.pdf(x, scale=data['sigma_fc'])
    elif xname == 'kc':
        return gamma_mustd(mu=data['mu_kc'], std=data['sigma_kc']).pdf(x)
    elif xname == 'Q':
        return gamma_mustd(mu=data['mu_Q'], std=data['sigma_Q']).pdf(x)
    elif xname == 'Pdet':
        return gamma_mustd(mu=data['mu_Pdet'], std=data['sigma_Pdet']).pdf(x)
    else:
        # Nothing to plot for logposterior
        return None

def range_from_pts(pts, percentile=1, pad=1.25):
    xmax = np.percentile(pts, 100-percentile)
    xmin = np.percentile(pts, percentile)
    xmid = (xmax + xmin) / 2.
    delta_x = (xmax - xmin) / 2.
    return xmid - delta_x * pad, xmid + delta_x * pad


def plot_all_traces(samp, data=None, prior_func=None):
    """
    Call prior_func(name, data, x) to get plot data.
    """
    N = len(samp)
    fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(N*1.5, N*1.5))
    for i, (yname, y) in enumerate(samp.items()):
        ymin, ymax = range_from_pts(y)
        for j, (xname, x) in enumerate(samp.items()):
            xmin, xmax = range_from_pts(x)
            ax = axes[i][j]
            if i != j:
                if i < j:
                    ax.plot(x, y, '.', markersize=4, alpha=0.1)
                else:
                    sns.kdeplot(x, y, ax=ax)
                ax.set_yticklabels([''])
                ax.set_xticklabels([''])
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                namey = AnchoredText(yname, 2, frameon=False)
                namex = AnchoredText(xname, 4, frameon=False)
                ax.add_artist(namey)
                ax.add_artist(namex)
            else:
                sns.kdeplot(x, ax=ax, legend=False)
                if prior_func is not None and data is not None:
                    line = ax.get_lines()[0]
                    xdata = line.get_xdata()
                    ydata = prior_func(xname, data, xdata)
                    if ydata is not None:
                        ax.plot(xdata, ydata)
                ax.set_yticklabels([''])
                ax.set_xticklabels([''])
                name = AnchoredText(xname, 2, frameon=False)
                ax.add_artist(name)

            if j == 0:
                ax.set_ylabel(yname)

            if i == len(samp.items())-1:
                ax.set_xlabel(xname)

    fig.tight_layout()
    return fig, axes


def HDI_from_MCMC(posterior_samples, credible_mass):
    """ Compute highest density interval from a sample of representative values,
    estimated as the shortest credible interval.
    
    Takes Arguments posterior_samples (samples from posterior),
    and credible mass (normally .95). 
    """
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0]*nCIs
    for i in range(nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
        HDImin = sorted_points[ciWidth.index(min(ciWidth))]
        HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
    return (HDImin, HDImax)

def sample_pymc3(d, samples=2000, njobs=2):
    with pm.Model() as model:
        dfc = pm.Normal(mu=0.0, sd=d['sigma_fc'], name='dfc')
        Q = pm.Gamma(mu=d['mu_Q'], sd=d['sigma_Q'], name='Q')
        Pdet = pm.Gamma(mu=d['mu_Pdet'], sd=d['sigma_Pdet'], name='Pdet')
        kc = pm.Gamma(mu=d['mu_kc'], sd=d['sigma_kc'], name='kc')

        M = d['M']
        T = d['T']
        scale=d['scale']
        mu_fc = d['mu_fc']
        f = d['f']


        like = pm.Gamma(alpha=M, beta=(M/(((2 * 1.381e-5 * T) / (np.pi * Q * kc)) / scale * (dfc + mu_fc)**3 /
                    ((f * f - (dfc + mu_fc)**2) * (f * f - (dfc + mu_fc)**2) + f * f * (dfc + mu_fc)**2 / Q**2)
                    + Pdet)),
                            observed=d['y'],
                            name='like')

        start = pm.find_MAP()
        step = pm.NUTS(state=start)
        
        trace = pm.sample(samples, step=step, start=start, progressbar=True, njobs=njobs)
    return trace


class PlotPyMCBrownian(object):
    def __init__(self, d, traces, name):

        self.data = d
        self.traces = traces
        self.name = name
        
        self.samples = OrderedDict()

        varnames = [var for var in traces.varnames if '_log_' not in var]
        for var in varnames:
            self.samples[var] = traces[var]

        self.y = self.data['y'] * self.data['scale']

        self.f = self.data['f']


        self.y_samp = eval_samples(Pf_func_gamma, self.f,
                                   self.samples, self.data)

        self.y_percent = lambda p: np.percentile(self.y_samp, p, axis=0)

        self.y_mean = np.mean(self.y_samp, axis=0)

        self.reduced_residuals = (self.y - self.y_mean) / self.y_mean

    def plot(self, fname=None, bbox_inches='tight', **kwargs):
        f = self.f

        Pdet = self.samples['Pdet'].mean() * self.data['scale']

        fig, ax = plt.subplots()
        ax.semilogy(self.f, self.y, color='b')
        ax.semilogy(self.f, self.y_mean, color='g')
        ax.semilogy(self.f, Pdet + np.zeros_like(self.f), 'g--')
        ax.semilogy(self.f, self.y_mean - Pdet, 'g--')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(u"PSD [nm²/Hz]")
        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, ax

    def plot_traces(self, fname=None, bbox_inches='tight', **kwargs):
        # Should plot the MCMC traces.
        axes = pm.traceplot(self.traces)
        fig = axes[0, 0].get_figure()
        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, axes


    def plot_zoomed(self, zoom=10, fname=None, bbox_inches='tight', **kwargs):
        f = self.f

        Delta_f  = (self.data['mu_fc'] / self.samples['Q'].mean()) * zoom
        mu_f = self.data['mu_fc'] + self.samples['dfc'].mean()
        m = (f >= (mu_f - Delta_f)) & (f <= (mu_f + Delta_f))

        fig, ax = plt.subplots()
        ax.semilogy(self.f[m], self.y[m], color='b')
        ax.semilogy(self.f[m], self.y_mean[m], color='g')
        ax.fill_between(self.f[m], self.y_percent(2.5)[m],
                        self.y_percent(97.5)[m], color='g', alpha=0.5)
        ax.set_xlim(mu_f - Delta_f, mu_f + Delta_f)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(u"PSD [nm²/Hz]")
        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, ax

    def plot_residuals(self, Hz_fraction=15, fname=None, bbox_inches='tight',
                        **kwargs):
        f_total = np.max(self.f) - np.min(self.f)
        frac = min(Hz_fraction / f_total, 1)
        self.residuals_lowess = lowess(self.reduced_residuals,
                                       self.f,
                                       frac=frac,
                                       return_sorted=False)
        
        fig, ax = plt.subplots()
        ax.plot(self.f, self.reduced_residuals, 'b.', alpha=0.5)
        ax.plot(self.f, self.residuals_lowess, 'g-', linewidth=2)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Reduced Residual")
        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)



    def plot_pairs(self, fname=None, bbox_inches='tight', **kwargs):
        fig, axes = plot_all_traces(self.samples, data=self.data,
                               prior_func=prior_func_pymc)

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, axes



    def summary(self, start=0, ndigits=4):
        trace = self.traces
        roundto = ndigits
        batches=100
        alpha=0.05

        varnames = [name for name in trace.varnames if not name.endswith('_')]

        stat_summ = pmstats._StatSummary(roundto, batches, alpha)
        pq_summ = pmstats._PosteriorQuantileSummary(roundto, alpha)
        summ = []
        for var in varnames:
            # Extract sampled values
            sample = trace.get_values(var, burn=start, combine=True)

            summ.append('\n%s:\n\n' % var)

            summ.append(stat_summ.output(sample))
            summ.append(pq_summ.output(sample))

        return '\n'.join(summ)



    def report(self, outfile=None, outdir=None, clean=True):
        if outfile is None:
            outfile = self.name

        basename = os.path.splitext(outfile)[0]

        if outdir is not None and not os.path.exists(outdir):
            os.mkdir(outdir)

        if outdir is None:
            outdir=''

        html_fname = os.path.join(outdir, basename+'.html')

        fit_fname = os.path.join(outdir, basename+'-fit.png')
        fit_zoomed_fname = os.path.join(outdir, basename+'-fit_zoomed.png')
        residuals_fname = os.path.join(outdir, basename+'-resid.png')
        pairs_fname = os.path.join(outdir, basename+'-pairs.png')
        traces_fname = os.path.join(outdir, basename+'-traces.png')


        self.plot(fname=fit_fname)
        self.plot_zoomed(fname=fit_zoomed_fname)
        self.plot_residuals(fname=residuals_fname)
        self.plot_pairs(fname=pairs_fname)
        self.plot_traces(fname=traces_fname)

        indented_summary_str = ['    '+line for 
                                line in self.summary().split('\n')]



        body = """\
======================
Brownian motion report
======================

Summary
=======

::

    {summary}

::

    fc [Hz]: {fc:.2f}
    Pdet [nm²/Hz]: {Pdet:.2e}


Fit
---

.. image:: {fit_fname}

Fit zoomed
----------

.. image:: {fit_zoomed_fname}


Residuals
---------

.. image:: {residuals_fname}


Pair Plots
----------

.. image:: {pairs_fname}

Trace Plot
----------

.. image:: {traces_fname}


""".format(summary='\n'.join(indented_summary_str),
           Pdet=self.samples['Pdet'].mean()*self.data['scale'],
           fc=self.samples['dfc'].mean()+self.data['mu_fc'],
           fit_fname=fit_fname,
           fit_zoomed_fname=fit_zoomed_fname,
           residuals_fname=residuals_fname,
           pairs_fname=pairs_fname,
           traces_fname=traces_fname)

        
        image_dependent_html = docutils.core.publish_string(body, writer_name='html')
        self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')

        with io.open(html_fname, 'w', encoding='utf8') as f:
            f.write(self_contained_html)

        if clean:
            for fname in [fit_fname, fit_zoomed_fname, 
                          residuals_fname, pairs_fname, traces_fname]:
            
                try:
                    os.remove(fname)
                except:
                    pass



def highlight_alternating(s, color='#EDEDED'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)

    is_max = 1 == (np.arange(len(s.index)) % 2)
    return pd.Series(np.where(is_max, attr, ''),
                            index=s.index)


# Note: Full workup should be a simple combination of,
# 1. Fitting
# 2. Plot / fit report

class PlotCmdStanBrownian(PlotPyMCBrownian):
    def __init__(self, d, traces, name):
        self.data = d
        self.traces = traces
        self.name = name
        
        self.samples = OrderedDict()

        for var in ['dfc', 'kc', 'Q', 'Pdet']:
            self.samples[var] = traces[var]

        self.y = self.data['y'] * self.data['scale']

        self.f = self.data['f']


        self.y_samp = eval_samples(Pf_func_gamma, self.f,
                                   self.samples, self.data)

        self.y_percent = lambda p: np.percentile(self.y_samp, p, axis=0)

        self.y_mean = np.mean(self.y_samp, axis=0)

        self.reduced_residuals = (self.y - self.y_mean) / self.y_mean

    def plot_pairs(self, fname=None, bbox_inches='tight', **kwargs):
        fig, axes = plot_all_traces(self.samples, data=self.data,
                               prior_func=prior_func_pymc)

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, axes

    # def plot_traces(self, fname=None, bbox_inches='tight', **kwargs):
    #     # Should plot the MCMC traces.
    #     axes = pm.traceplot(self.traces)
    #     fig = axes[0, 0].get_figure()
    #     if fname is not None:
    #         fig.tight_layout()
    #         fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

    #     return fig, axes

    def report(self, outfile=None, outdir=None, clean=True):
        if outfile is None:
            outfile = self.name

        basename = os.path.splitext(outfile)[0]

        if outdir is not None and not os.path.exists(outdir):
            os.mkdir(outdir)

        if outdir is None:
            outdir=''

        html_fname = os.path.join(outdir, basename+'.html')

        fit_fname = os.path.join(outdir, basename+'-fit.png')
        fit_zoomed_fname = os.path.join(outdir, basename+'-fit_zoomed.png')
        residuals_fname = os.path.join(outdir, basename+'-resid.png')
        pairs_fname = os.path.join(outdir, basename+'-pairs.png')
        traces_fname = os.path.join(outdir, basename+'-traces.png')

        self.plot(fname=fit_fname)
        self.plot_zoomed(fname=fit_zoomed_fname)
        self.plot_residuals(fname=residuals_fname)
        self.plot_pairs(fname=pairs_fname)
        # self.plot_traces(fname=traces_fname)

        self.traces['fc'] = self.traces['dfc'] + self.data['mu_fc']
        self.traces['Gamma'] = self.traces['kc'] / (2*np.pi*self.traces['fc']*self.traces['Q'])
        self.traces['Pdet_nm'] = self.traces['Pdet']  * self.data['scale']

        traces = self.traces[['fc', 'kc', 'Q', 'Pdet_nm', 'Gamma', 'lp__', 'accept_stat__', 'stepsize__', 'treedepth__',
       'n_leapfrog__', 'divergent__', 'energy__', 'dfc', 'Pdet']]

        describe = traces.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        text = describe.style.apply(highlight_alternating, axis=0).render()
        lines = ["    "+line for line in text.splitlines()]
        summary_txt = "\n".join(lines)

        body = """\
======================
Brownian motion report
======================

Summary
=======

 .. raw:: html

 {summary}

::

    fc [Hz]: {fc:.2f}
    Pdet [nm²/Hz]: {Pdet:.2e}


Fit
---

.. image:: {fit_fname}

Fit zoomed
----------

.. image:: {fit_zoomed_fname}


Residuals
---------

.. image:: {residuals_fname}


Pair Plots
----------

.. image:: {pairs_fname}


""".format(summary=summary_txt,
           Pdet=self.samples['Pdet'].mean()*self.data['scale'],
           fc=self.samples['dfc'].mean()+self.data['mu_fc'],
           fit_fname=fit_fname,
           fit_zoomed_fname=fit_zoomed_fname,
           residuals_fname=residuals_fname,
           pairs_fname=pairs_fname)

        
        image_dependent_html = docutils.core.publish_string(body, writer_name='html')
        self_contained_html = unicode(img2uri(image_dependent_html), 'utf8')

        with io.open(html_fname, 'w', encoding='utf8') as f:
            f.write(self_contained_html)

        if clean:
            for fname in [fit_fname, fit_zoomed_fname, 
                          residuals_fname, pairs_fname, traces_fname]:
            
                try:
                    os.remove(fname)
                except:
                    pass




@click.command(help="""\
Fit brownian motion data using Bayesian MCMC, using data from an HDF5 file.

Arguments:

\b
FILENAME            HDF5 datafile
FMIN                Min frequency to fit [Hz]
FMAX                Max frequency to fit [Hz]
""")
@click.argument('filename', type=click.Path(exists=True))
@click.argument('fmin', type=float)
@click.argument('fmax', type=float)
@click.option('--output', '-o', type=str, default=None)
@click.option('--temperature', '-T', default=298., help='Temperature (298) [K]')
@click.option('--spring-constant', '-k', default=3.5, help='Spring constant prior mean (3.5 N/m)')
@click.option('--quality-factor', '-Q', default=20000., help='Quality factor prior mean (20000)')
@click.option('--spring-constant-stdev', '-sk', default=5., help='Spring constant prior standard deviation (5 N/m)')
@click.option('--quality-factor-stdev', '-sQ', default=20000., help='Quality factor prior standard deviation (20000)')
@click.option('--resonance-frequency-stdev', '-sf', default=5., help='Resonance frequency prior standard deviation')
@click.option('--detector-noise', '-P', default=None, help='Detector noise prior mean (None, est. from data)')
@click.option('--detector-noise-stdev', '-sP', default=None, help='Detector noise prior standard deviation (None, est. from data)')
@click.option('--chains', '-c', default=2, help="MCMC chains to run (2)")
@click.option('--iterations', '-i', default=2000, help='MCMC iterations per chain (2000)')
@click.option('--clean/--no-clean', default=True, help='Clean intermediate png files.')
def pymc_brownian_cli(filename, fmin, fmax, output, spring_constant,
                          quality_factor, temperature, spring_constant_stdev,
                          quality_factor_stdev, resonance_frequency_stdev,
                          detector_noise, detector_noise_stdev,
                          chains, iterations, clean):
    kc = spring_constant
    Q = quality_factor
    T = temperature
    sigma_kc = spring_constant_stdev
    sigma_Q = quality_factor_stdev
    sigma_fc = resonance_frequency_stdev
    Pdet = detector_noise
    sigma_Pdet = detector_noise_stdev
    fh = h5py.File(filename, 'r')
    data = fh2data(fh, fmin, fmax, kc, Q, sigma_Q=sigma_Q, T=T, sigma_kc=sigma_kc,
                   sigma_fc=sigma_fc, Pdet=Pdet, sigma_Pdet=sigma_Pdet)

    traces = sample_pymc3(data, iterations, chains)

    name = os.path.splitext(filename)[0]

    pbb = PlotPyMCBrownian(data, traces, name)

    if output is None:
        output = os.path.splitext(filename)[0]+'.html'

    pbb.report(output, clean=clean)


@txt_filename
def split_header_data(fh):
    comments = []
    data = []
    for line in fh:
        if line.startswith('#'):
            comments.append(line)
        else:
            data.append(line)

    return (data, comments)

def cmdstan_sample(data, iterations, chains, warmup=None,
                   model=directory+'/stanmodels/gamma'+'.'+script_extension,
                   modelname='gamma', datafile='data.dump'):
    with open(datafile, 'w') as f:
        dump_to_rdata(f, **data)

    if warmup is None:
        warmup = iterations

    processes = [subprocess.Popen([model, 'sample', 'num_samples={}'.format(iterations),
                      'num_warmup={}'.format(warmup),
                      'data', 'file={}'.format(datafile),
                      'id={}'.format(i),
                      'output',
                      'file={}-{}.csv'.format(modelname, i)])
                for i in xrange(1, chains+1)]

    ls = [psutil.Process(p.pid) for p in processes]

    gone, alive = psutil.wait_procs(ls, timeout=1000)

    filenames = ['{}-{}.csv'.format(modelname, i)
                for i in xrange(1, chains+1)]

    files = [open(filename, 'r') for filename in filenames]

    for i, fh in enumerate(files):
        data, comments = split_header_data(fh)
        data_io = StringIO("".join(data))
        df = pd.read_csv(data_io)
        if i == 0:
            df_full = df
        else:
            df_full = df_full.append(df, ignore_index=True)

    return df_full





@click.command(help="""\
Fit brownian motion data using Bayesian MCMC, using data from an HDF5 file.

Arguments:

\b
FILENAME            HDF5 datafile
FMIN                Min frequency to fit [Hz]
FMAX                Max frequency to fit [Hz]
""")
@click.argument('filename', type=click.Path(exists=True))
@click.argument('fmin', type=float)
@click.argument('fmax', type=float)
@click.option('--output', '-o', type=str, default=None)
@click.option('--temperature', '-T', default=298., help='Temperature (298) [K]')
@click.option('--spring-constant', '-k', default=3.5, help='Spring constant prior mean (3.5 N/m)')
@click.option('--quality-factor', '-Q', default=20000., help='Quality factor prior mean (20000)')
@click.option('--spring-constant-stdev', '-sk', default=5., help='Spring constant prior standard deviation (5 N/m)')
@click.option('--quality-factor-stdev', '-sQ', default=20000., help='Quality factor prior standard deviation (20000)')
@click.option('--resonance-frequency-stdev', '-sf', default=5., help='Resonance frequency prior standard deviation')
@click.option('--detector-noise', '-P', default=None, help='Detector noise prior mean (None, est. from data)')
@click.option('--detector-noise-stdev', '-sP', default=None, help='Detector noise prior standard deviation (None, est. from data)')
@click.option('--chains', '-c', default=2, help="MCMC chains to run (2)")
@click.option('--iterations', '-i', default=2000, help='MCMC iterations per chain (2000)')
@click.option('--clean/--no-clean', default=True, help='Clean intermediate png files.')
def cmdstan_brownian_cli(filename, fmin, fmax, output, spring_constant,
                          quality_factor, temperature, spring_constant_stdev,
                          quality_factor_stdev, resonance_frequency_stdev,
                          detector_noise, detector_noise_stdev,
                          chains, iterations, clean):
    kc = spring_constant
    Q = quality_factor
    T = temperature
    sigma_kc = spring_constant_stdev
    sigma_Q = quality_factor_stdev
    sigma_fc = resonance_frequency_stdev
    Pdet = detector_noise
    sigma_Pdet = detector_noise_stdev
    fh = h5py.File(filename, 'r')
    data = fh2data(fh, fmin, fmax, kc, Q, sigma_Q=sigma_Q, T=T, sigma_kc=sigma_kc,
                   sigma_fc=sigma_fc, Pdet=Pdet, sigma_Pdet=sigma_Pdet)

    # To switch to cmdstan, I need to:
    # Translate data to a rdata.dump file
    # Run subprocess command to sample
    # Collect data from .csv
    # Merge chains if necessary
    # 

    # To switch to cmdstan, I need to:
    # Translate data to a rdata.dump file
    # Run subprocess command to sample
    # Collect data from .csv
    # Merge chains if necessary
    # 
    traces = cmdstan_sample(data, iterations, chains)

    name = os.path.splitext(filename)[0]

    pbb = PlotCmdStanBrownian(data, traces, name)

    if output is None:
        output = os.path.splitext(filename)[0]+'.html'

    pbb.report(output, clean=clean)


