# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
import os
import io

import datetime
import copy
import platform
import pystan
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
from brownian import u, calc_P_x0, Pf
sns.set_style("white")
from matplotlib.offsetbox import AnchoredText
from six.moves import cPickle as pickle
from statsmodels.nonparametric.smoothers_lowess import lowess
from six import string_types
from pystan.misc import _array_to_table
from brownian._calck import img2uri
from pystan.external.pymc import plots

windows = 'Windows' == platform.system()

directory = os.path.split(__file__)[0]
model_code_dict_fname = os.path.join(directory, 'stan_model_code.pkl')

def prnDict(aDict, br='\n', html=0,
            keyAlign='l',   sortKey=0,
            keyPrefix='',   keySuffix='',
            valuePrefix='', valueSuffix='',
            leftMargin=4,   indent=1, braces=True):
    '''
return a string representive of aDict in the following format:
    {
     key1: value1,
     key2: value2,
     ...
     }

Spaces will be added to the keys to make them have same width.

sortKey: set to 1 if want keys sorted;
keyAlign: either 'l' or 'r', for left, right align, respectively.
keyPrefix, keySuffix, valuePrefix, valueSuffix: The prefix and
   suffix to wrap the keys or values. Good for formatting them
   for html document(for example, keyPrefix='<b>', keySuffix='</b>'). 
   Note: The keys will be padded with spaces to have them
         equally-wide. The pre- and suffix will be added OUTSIDE
         the entire width.
html: if set to 1, all spaces will be replaced with '&nbsp;', and
      the entire output will be wrapped with '<code>' and '</code>'.
br: determine the carriage return. If html, it is suggested to set
    br to '<br>'. If you want the html source code eazy to read,
    set br to '<br>\n'

version: 04b52
author : Runsun Pan
require: odict() # an ordered dict, if you want the keys sorted.
         Dave Benjamin 
         http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/161403
    '''
   
    if aDict:

        #------------------------------ sort key
        if sortKey:
            dic = aDict.copy()
            keys = dic.keys()
            keys.sort()
            aDict = odict()
            for k in keys:
                aDict[k] = dic[k]

        #------------------- wrap keys with ' ' (quotes) if str
        tmp = ['{']
        ks = [type(x)==str and "'%s'"%x or x for x in aDict.keys()]

        #------------------- wrap values with ' ' (quotes) if str
        vs = [type(x)==str and "'%s'"%x or x for x in aDict.values()] 

        maxKeyLen = max([len(str(x)) for x in ks])

        for i in range(len(ks)):

            #-------------------------- Adjust key width
            k = {1            : str(ks[i]).ljust(maxKeyLen),
                 keyAlign=='r': str(ks[i]).rjust(maxKeyLen) }[1]

            v = vs[i]
            tmp.append(' '* indent+ '%s%s%s:%s%s%s,' %(
                        keyPrefix, k, keySuffix,
                        valuePrefix,v,valueSuffix))

        tmp[-1] = tmp[-1][:-1] # remove the ',' in the last item
        tmp.append('}')

        if leftMargin:
          tmp = [ ' '*leftMargin + x for x in tmp ]

        if not braces:
            tmp = tmp[5:-2]

        if html:
            return '<code>%s</code>' %br.join(tmp).replace(' ','&nbsp;')
        else:
            return br.join(tmp)
    else:
        return '{}'

def memodict(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret 
    return memodict().__getitem__

# Need a dictionary of just, model_name, model_code pickled.

def model_pkl_file(model_name):
    return os.path.join(directory, 'stanmodels/', model_name+'.pkl')


def update_models_dict(model_code_dict, old_model_code_dict={}, test=False):
    """Compile outdated stan models and return in a dictionary for pickling.

    Models are recompiled (which can take 10-20 seconds) if the model code has
    changed."""
    updated_model_code_dict = {}
    for model_name, model_code in model_code_dict.items():
        if model_name not in old_model_code_dict or model_code != old_model_code_dict[model_name]:
            # test = True bypasses time-intensive compilation,
            # so this function can be tested quickly.
            updated_model_code_dict[model_name] = (model_code)
            if not test:
                sm = pystan.StanModel(model_code=model_code,
                                                  model_name=model_name)
                pickle.dump(sm, open(model_pkl_file(model_name), 'wb'))
        else:
            updated_model_code_dict[model_name] = old_model_code_dict[model_name]

    return updated_model_code_dict


def Pfi(d, samp, i):
    fc = d['mu_fc'] + samp['dfc'][i]
    Q = samp['Q'][i]
    kc = samp['kc'][i]
    return Pf(d['f'],
              calc_P_x0(fc*u.Hz, Q, kc*u('N/m'), d['T']*u.K).to('nm^2/Hz').magnitude,
              fc, Q, samp['Pdet'][i]*d['scale']
             )

# def Pf(x, samp, d):
#     """The equation to calculate :math:`P_x(f)` from the cantilever
#     parameters and detector noise floor.

#     f
#         Frequency, the independent variable.

#     P_x0
#         The zero frequency power spectral density of position fluctuations.

#     f_c
#         The cantilever resonance frequency.

#     Q
#         The quality factor of the cantilever.

#     P_detector
#         The detector noise floor.
#         """
#     fc = d['mu_fc'] + samp['dfc'][i]
#     Q = samp['Q'][i]
#     kc = samp['kc'][i]
#     return Pf(x,
#               calc_P_x0(fc*u.Hz, Q, kc * u('N/m'), T*u.K).to('nm^2/Hz').magnitude,
#               fc, Q, samp['Pdet'] * d['scale']
#              )

# Would like to arbitrarily vectorize over samples. Function takes x, samples, data.

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


model_code_dict = {
'gamma': """
data {
  int<lower=0> N;
  int<lower=0> M;
  vector[N] f;
  vector[N] y;
  real mu_fc;
  real mu_kc;
  real mu_Q;
  real mu_Pdet;
  real sigma_fc;
  real sigma_kc;
  real sigma_Q;
  real sigma_Pdet;
  real scale;
  real<lower=0> T;
}
parameters {
  real dfc;
  real<lower=0> kc;
  real<lower=0> Q;
  real<lower=0> Pdet;
}
model {
    # Priors on fit parameters
    dfc ~ normal(0, sigma_fc);
    kc ~ normal(mu_kc, sigma_kc);
    Q ~ normal(mu_Q, sigma_Q);
    Pdet ~ cauchy(mu_Pdet, sigma_Pdet);
    
    y ~ gamma(M, M ./ (
    ((2 * 1.381e-5 * T) / (pi() * Q * kc)) / scale * (dfc + mu_fc)^3 ./
            ((f .* f - (dfc + mu_fc)^2) .* (f .* f - (dfc + mu_fc)^2) + f .* f * (dfc + mu_fc)^2 / Q^2)
            + Pdet)
            );
}
"""
}

def update_models(model_code_dict=model_code_dict,
                  stanmodel_pkl_file=model_code_dict_fname,
                  recompile_all=False,
                  global_model=True):
    """Update stan models if model code has changed. Otherwise load from disk.

    Setting recompile_all=True forces all models to be recompiled."""
    existing_dict = {}
    if not recompile_all:
        try:
            existing_dict = pickle.load(open(stanmodel_pkl_file, 'rb'))
        except IOError:
            pass

    if global_model:
        global models
    models = update_models_dict(model_code_dict, existing_dict)

    pickle.dump(models, open(stanmodel_pkl_file, 'wb'))

    return models


update_models()

pickle.dump(model_code_dict, open(model_code_dict_fname, 'wb'))

@memodict
def get_model(model_name):
    return pickle.load(open(model_pkl_file(model_name), 'rb'))


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



def initial(d, k0=0.6, Pdet=None):
    if Pdet is None:
        Pdet = d['sigma_Pdet']*0.1
    return lambda: {'dfc': 0, 'kc': d['mu_fc'],
                    'Q': d['mu_Q'], 'Pdet': Pdet, 'k': k0}


def halfnorm_stan(x, loc=0, scale=1, lower=0):
    total = 1 - stats.norm.cdf(lower, scale=scale, loc=loc)
    return np.where(x >= lower, stats.norm.pdf(x, loc=loc, scale=scale), 0.)


def halfcauchy_stan(x, loc=0, scale=1, lower=0.):
    total = 1 - stats.cauchy.cdf(lower, scale=scale, loc=loc)
    return np.where(x >= lower, stats.cauchy.pdf(x, loc=loc, scale=scale), 0.)


def prior_func_gamma(xname, data, x):
    if xname == 'dfc':
        return norm.pdf(x, scale=data['sigma_fc'])
    elif xname == 'kc':
        return halfnorm_stan(x, loc=data['mu_kc'], scale=data['sigma_kc'])
    elif xname == 'Q':
        return halfnorm_stan(x, loc=data['mu_Q'], scale=data['sigma_Q'])
    elif xname == 'Pdet':
        return halfcauchy_stan(x, loc=data['mu_Pdet'], scale=data['sigma_Pdet'])
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
                sns.kdeplot(x, ax=ax)
                if prior_func is not None and data is not None:
                    line = ax.get_lines()[0]
                    xdata = line.get_xdata()
                    ydata = prior_func(xname, data, xdata)
                    if ydata is not None:
                        ax.plot(xdata, ydata)
                ax.set_yticklabels([''])
                ax.set_xticklabels([''])
                name = AnchoredText(xname,2,  frameon=False)
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


def save(gr, model_name, model_code, out, compress=True):
    if compress:
        kwargs = {'compression': "gzip", 'compression_opts':9, 'shuffle':True}
    else:
        kwargs = {}

    gr['model_name'] = model_name
    gr['model_code'] = model_code
    gr['timestamp'] = datetime.datetime.isoformat(datetime.datetime.now())

    summary = out.summary()
    params = out.extract(permuted=True)

    gr.create_dataset('summary', summary['summary'].shape, **kwargs)
    gr['summary'][:] = summary['summary']

    gr['summary_colnames'] = np.array(summary['summary_colnames'], dtype=np.str)
    gr['summary_rownames'] = np.array(summary['summary_rownames'], dtype=np.str)
    param_gr = gr.create_group('params')
    for key, val in params.items():
        param_gr.create_dataset(key, val.shape, **kwargs)
        param_gr[key][:] = val

    gr['parameters'] = np.array(params.keys(), dtype=np.str)

    data_gr = gr.create_group('data')
    for key, val in out.data.items():
        # Try to create a compressed dataset; if it fails, do a 
        try:
            data_gr.create_dataset(key, val.shape, **kwargs)
            data_gr[key][:] = val
        except (AttributeError, TypeError):
            data_gr[key] = val


def gr2datadict(gr):
    return {key: val.value for key, val in gr['data'].items() if not isinstance(val.value, np.ndarray)}


def gr2summary_str(gr, ndigits=2):
    summary = [
        gr['model_name'].value,
        gr.file.filename,
        prnDict(gr2datadict(gr), braces=False),
        _array_to_table(gr['summary'][:], gr['summary_rownames'][:], 
                gr['summary_colnames'][:], ndigits)
    ]
    return '\n\n'.join(summary)


# This is the *exact* same code!
class BayesianBrownian(object):
    """Class for Bayesian Brownian motion fitting."""
    def __init__(self, model_name, data_or_fname, model=None, priors=None):
        self.model_name = model_name
        self.model_code = models[model_name]

        if model is None:
            self.sm = get_model(model_name)
        else:
            self.sm = model

        if isinstance(data_or_fname, string_types):
            self.data = fh2data(data_or_fname)
            self.data_fname = data_or_fname
        else:
            self.data = data_or_fname

        
        if priors is None:
            self.default_priors = default_priors[model_name]
            self.priors = copy.copy(self.default_priors)
        else:
            self.priors = priors

    def sample(self, chains=4, iter=2000, priors=None, init='random',
               **kwargs):
        if priors is not None:
            self.priors = priors
        updated_data = copy.copy(self.data)
        updated_data.update(self.priors)
        # See http://pystan.readthedocs.org/en/latest/windows.html#windows
        # Must specify njobs=1 on windows
        if windows:
            kwargs['njobs'] = 1
        self.out = self.sm.sampling(data=updated_data, chains=chains, iter=iter,
                                    init=init, **kwargs)
        self.samp = self.out.extract()
        return self.out

    def optimize(self, priors=None, init='random', **priors_kwargs):
        if priors is not None:
            self.priors = priors
        self.priors.update(priors_kwargs)
        updated_data = copy.copy(self.data)
        updated_data.update(self.priors)
        self.popt = self.sm.optimizing(data=updated_data, init=init)
        return self.popt

    def save(self, gr_or_fname, compress=True):
        if isinstance(gr_or_fname, string_types):
            with h5py.File(gr_or_fname, 'w') as gr:
                save(gr, self.model_name, self.model_code, self.out,
                     compress=compress)
        else:
            save(gr_or_fname, self.model_name, self.model_code, self.out,
                 compress=compress)


class PlotBayesianBrownian(object):
    def __init__(self, fh_or_fname):
        if isinstance(fh_or_fname, string_types):
            self.fh = h5py.File(fh_or_fname, 'r')
            self.filename = fh_or_fname
        else:
            self.fh = fh_or_fname
            self.filename = self.fh.filename

        self.data = {key: val.value for key, val in self.fh['data'].items()}
        
        self.samples = OrderedDict()
        for key, val in self.fh['params'].items():
            self.samples[key] = val[:]

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

    def plot_chains(self):
        # Should plot the MCMC traces.
        pass

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
                               prior_func=prior_func_gamma)

        if fname is not None:
            fig.tight_layout()
            fig.savefig(fname, bbox_inches=bbox_inches, **kwargs)

        return fig, axes



    def summary(self, ndigits=3):
        return gr2summary_str(self.fh, ndigits=ndigits)

    def report(self, outfile=None, outdir=None):
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


        self.plot(fname=fit_fname)
        self.plot_zoomed(fname=fit_zoomed_fname)
        self.plot_residuals(fname=residuals_fname)
        self.plot_pairs(fname=pairs_fname)

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


""".format(summary='\n'.join(indented_summary_str),
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

        for fname in [fit_fname, fit_zoomed_fname, 
                      residuals_fname, pairs_fname]:
        
            try:
                os.remove(fname)
            except:
                pass


# This looks pretty good. Let's add information on fh (dataset, etc),
# and write a cli script.

# Note: Full workup should be a simple combination of,
# 1. Fitting
# 2. Plot / fit report

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
@click.option('--chains', '-c', default=4, help="MCMC chains to run (4)")
@click.option('--iterations', '-i', default=1000, help='MCMC iterations per chain')
def bayesian_brownian_cli(filename, fmin, fmax, output, spring_constant,
                          quality_factor, temperature, spring_constant_stdev,
                          quality_factor_stdev, resonance_frequency_stdev,
                          chains, iterations):
    kc = spring_constant
    Q = quality_factor
    T = temperature
    sigma_kc = spring_constant_stdev
    sigma_Q = quality_factor_stdev
    sigma_fc = resonance_frequency_stdev
    fh = h5py.File(filename, 'r')
    data = fh2data(fh, fmin, fmax, kc, Q, sigma_Q=sigma_Q, T=T, sigma_kc=sigma_kc,
                   sigma_fc=sigma_fc)

    # Need an option to use a real file here.
    fhsamp = h5py.File('.inmem.h5', driver='core', backing_store=False)

    b = BayesianBrownian('gamma', data, priors={})

    b.sample(chains=chains, iter=iterations)
    b.save(fhsamp)
    pbb = PlotBayesianBrownian(fhsamp)
    if output is None:
        output = os.path.splitext(filename)[0]+'.html'
    pbb.report(output)

