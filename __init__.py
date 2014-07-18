"""
Noise Fitting Brownian Motion Data
Ryan Dwyer
2013-11-22

This document will implement John Marohn's Brownian motion noise 
fitting routine in Python. The file has the following stuructre.

    class BrownianMotionFitter
        This is the main object that stores, fits, and plots the data.


Dependencies:
numpy
scipy
matplotlib
pint
pandas
pytables

"""

from __future__ import division
import numpy as np
import scipy
import scipy.stats
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
# Backend should be set correctly here.
from uncertainties import correlated_values, ufloat
import h5py
from jittermodel.ubase import u
from jittermodel.ubase import UnitCantilever

k_B = 1.3806504e-23 * u.J / u.K


class BrownianMotionFitter(object):
    """This object fits Brownian motion position fluctuation data to
    calculate the spring constant of the cantilever and the
    detector noise floor.
    """

    def __init__(self, f, PSD, PSD_ci, T, est_cant):
        """
        f
            frequency (Hz)

        PSD
            Power spectral density of position fluctuations (nm^2 / Hz)

        PSD_ci
            The confidence intervals corresponding to the PSD data (nm^2 / Hz)

        T
            temperature (K)

        est_cant
            A unitted cantilever.

        """
        self.f = f
        self.PSD_raw = PSD
        self.PSD_ci_raw = PSD_ci
        self.T = T * u.K
        self.est_cant = est_cant

    def _guess_P_detector(self):
        """Guess a low, but reasonable number for the detector noise floor.
        This is used to scale the data for fitting."""
        self.P_detector0_raw = np.percentile(self.PSD_raw, 2)

    def _scale_data(self):
        """Scale the raw data for fitting, so that floating point number errors
        are not an issue."""
        self.PSD = self.PSD_raw / self.P_detector0_raw
        self.P_detector0 = 1

    def calc_fit(self, f_min=1e3, f_max=1e5):
        """Fit the power spectrum data over the frequency range
        f_min to f_min, using a 3 pass approach."""
        self.f_min = f_min
        self.f_max = f_max

        self.mask = mask = make_mask(self.f, f_min, f_max)

        self._guess_P_detector()
        self._scale_data()

        f = self.fit_f = self.f[mask]
        PSD = self.fit_PSD = self.PSD[mask]
        PSD_ci = self.PSD_ci = self.PSD_ci_raw[mask]
        self.fit_PSD_raw = self.PSD_raw[mask]

        self.calc_initial_params()

        self._first_pass(f, PSD, PSD_ci)
        self._second_pass(f, PSD, PSD_ci)
        self._final_pass(f, PSD, PSD_ci)
        self._prepare_output()

    def calc_initial_params(self):
        """Use the estimates to calculate initial parameters to use for
        the fitting functions."""
        f_c = self.est_cant.f_c.to(u.Hz)
        k_c = self.est_cant.k_c.to(u.N/u.m)
        Q = self.est_cant.Q.magnitude

        P_x0guess = (calc_P_x0(f_c, k_c, Q, self.T) /
                    (self.P_detector0_raw * u.nm ** 2 / u.Hz))

        self.initial_params = [P_x0guess.magnitude, f_c.magnitude, Q]

    def _first_pass(self, f, PSD, PSD_ci):
        """Fit the power spectrum data using initial guesses for the
        parameters and a fixed noise floor. Fitting outputs stored in
        self.popt1, self.pcov1.

        We use user input guesses for the three cantilever parameters,
        and fix the thermal noise floor to a low but reasonable guess.
        Then, we fit the remaining three parameters :math:`P_{x0}, f_c, Q`.
        """

        def Pf_fixed_P_detector(f, P_x0, f_c, Q):
            return Pf(f, P_x0, f_c, Q, self.P_detector0)

        self.popt1, self.pcov1 = curve_fit(Pf_fixed_P_detector,
                                           f, PSD,
                                           p0=self.initial_params,
                                           sigma=PSD_ci)

    def _second_pass(self, f, PSD, PSD_ci):
        """Fit the power spectrum holding the center frequency :math:`f_c`
        fixed. Fitting outputs stored in self.popt2, self.pcov2."""
        f_c = self.popt1[1]

        def Pf_fixed_f_c(f, P_x0, Q, P_detector):
            return Pf(f, P_x0, f_c, Q, P_detector)

        p0 = [self.popt1[0], self.popt1[2], self.P_detector0]

        self.popt2, self.pcov2 = curve_fit(Pf_fixed_f_c,
                                           f, PSD,
                                           p0=p0,
                                           sigma=PSD_ci)

    def _final_pass(self, f, PSD, PSD_ci):
        """Fit the power spectrum allowing all parameters to
        determined by the fitting algorithm. Fitting outputs stored to
        self.popt, self.pcov.

        The fit data and raw fit data is stored to self.PSD_fit,
        self.PSD_fit_raw."""
        popt1, popt2 = self.popt1, self.popt2
        f_c = popt1[1]
        p0 = [popt2[0], f_c, popt2[1], popt2[2]]
        self.popt, self.pcov = curve_fit(Pf,
                                         f, PSD,
                                         p0=p0,
                                         sigma=PSD_ci)
        self.PSD_fit = Pf(f, *self.popt)
        self.PSD_fit_raw = self.PSD_fit * self.P_detector0_raw

    def _prepare_output(self):
        f_c, k_c, Q, P_detector = translate_fit_parameters(
            self.popt, self.pcov, self.P_detector0_raw, self.T)
        self.residuals = (self.PSD_fit - self.PSD[self.mask])
        self.reduced_residuals = self.residuals / self.PSD_fit
        self.reduced_residuals_sorted = np.sort(self.reduced_residuals)
        self.f_c, self.k_c, self.Q, self.P_detector = f_c, k_c, Q, P_detector

    def print_output(self):
        print("""
    Resonence frequency f_c: {self.f_c:!p}
    Spring constant k_c: {self.k_c:!p}
    Quality Factor Q: {self.Q:!p}
    Detector Noise: {self.P_detector}
            """.format(self=self))

    def plot_fit(self):
        """Plot the calculated fit."""
        f = self.fit_f
        plt.semilogy(f, self.fit_PSD_raw, f, self.PSD_fit_raw)
        plt.xlim(self.f_min, self.f_max)
        plt.show()

    def plot_residuals(self):
        """Plot the residuals of the calculated fit."""
        plt.plot(self.fit_f, self.residuals)
        plt.show()

    def plot_reduced_residuals(self):
        plt.plot(self.fit_f, self.reduced_residuals)
        plt.show()

    def fit_residuals(self):
        self.p_residuals = fit_residuals(self.reduced_residuals_sorted)
        print("""The residuals have mean {self.p_residuals[0]:.2e}
and standard deviation {self.p_residuals[1]:.2e}""".format(self=self))


def Pf(f, P_x0, f_c, Q, P_detector):
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
    return (P_x0 * f_c**4 /
            ((f**2 - f_c**2)**2 + f**2 * f_c**2 / Q**2)
            + P_detector)


def make_mask(f, f_min, f_max):
    """Return a mask of the array f. The mask can be used
    to give only elements of f between fmin and fmax.

    Use the resulting mask to block off elements of f outside
    of the specified bounds, as shown below.

    >>> f = np.arange(10)
    >>> mask = make_mask(f, 1.5, 5.5)
    >>> f[mask]
    array([2, 3, 4, 5])
    """
    mask_low = f < f_max
    mask_high = f > f_min
    return np.logical_and(mask_high, mask_low)


def translate_fit_parameters(popt, pcov, P_detector0_raw, T=300*u.K):
    """Take the fit parameters and covariances, and converts them to
    SI values and errors for f_c, k_c, Q."""
    pvals = correlated_values(popt, pcov)
    punits = [u.nm**2 / u.Hz, u.Hz, u.dimensionless, u.nm**2 / u.Hz]
    scales = [P_detector0_raw, 1, 1, P_detector0_raw]

    P_x0, f_c, Q, P_detector = [uncert_val * unit * scale for
                                uncert_val, unit, scale in
                                zip(pvals, punits, scales)]

    k_c = calc_k_c(f_c, Q, P_x0, T)
    return f_c, k_c, Q, P_detector


def calc_k_c(f_c, Q, P_x0, T=ufloat(300, 1)*u.K):
    """Calculate the spring constant, returning a pint quantity containing
    a value with uncertainty, if uncertainty is used on the input (ufloat)."""
    return ((2 * k_B * T) / (np.pi * f_c * Q * P_x0)).to(u.N / u.m)


def calc_P_x0(f_c, Q, k_c, T):
    """Calculate the thermal noise floor, returning a pint quantity in
    units of nm^2 / Hz."""
    return ((2 * k_B * T) / (np.pi * f_c * Q * k_c)).to(u.nm ** 2 / u.Hz)


def iterate_fit(func, x, y, p0, sigma_i, wgt):
    popt = [[]]
    pcov = [[]]
    popt[0], pcov[0] = curve_fit(func, x, y, p0=p0, sigma=sigma_i)
    for i in xrange(10):
        new_sigma = wgt * func(x, *popt[-1])
        opt, cov = curve_fit(func, x, y, p0=popt[-1], sigma=new_sigma)
        popt.append(opt)
        pcov.append(cov)
    PSD_fit = func(x, *popt[-1])
    return popt[-1], pcov[-1], PSD_fit


def avg_ci_data(data, axis=0):
    """Average the data along the given dimension, and calculate the 95
    percent confidence interval. Return the avg, ci arrays."""
    data_avg = data.mean(axis=axis)
    data_std = data.std(axis=axis, ddof=1)
    data_ci = data_std / data.shape[axis]**0.5 * 1.96
    return data_avg, data_ci


def get_data(filename):
    """Extract power spectrum data from an HDF5 file.
    Return f, PSD_mean, PSD_conf_int.

    The HDF file should have the following structure:

    f
        A table containing the frequencies of each point in the power
        spectrum.

    PSD
        Contains power spectra PSD_0(f) through PSD_(n-1)(f), arranged
            as follows, with each power spectrum occupying 1 column. 
        PSD_0(0 Hz) PSD_1(O Hz)
        PSD_0(1 Hz) PSD_1(1 Hz) ...


    See http://h5labview.sourceforge.net and http://pytables.github.io for
    more information."""
    with h5py.File(filename, 'r') as fh:
        if fh.get('PSD', getclass=True) is h5py.Group:
            raise ValueError("It appears you have inputted an old hdf5 file,\
                              with power spectrum in a separate dataset.\
                              Use the function convert_data to convert to\
                              the new file format, and try getting the\
                              data again.")
        f = fh['f'].value
        PSD = fh['PSD'].value

    PSD_mean, PSD_ci = avg_ci_data(PSD, axis=1)

    return f, PSD_mean, PSD_ci


def fit_residuals(sorted_residuals):
    """Return the result of fitting the residuals to the normal cdf.
    """
    size = sorted_residuals.size
    y = np.arange(1, 1 + size) / size

    def cdf(x, loc, scale):
        return scipy.stats.norm.cdf(x, loc=loc, scale=scale)

    popt, _ = curve_fit(cdf, sorted_residuals, y)

    return popt


def convert_data(oldfile, newfile, fileformat='v1'):
    """Converts data from the v1 format to the v2 format"""
    oldf = h5py.File(oldfile, 'r')
    if oldf.get('PSD', getclass=True) is not h5py.Group:
        oldf.close()
        raise ValueError("It appears you have inputted a new style file.\
            The file will not be converted.")
    else:
        newf = h5py.File(newfile, 'w')
        newf['f'] = oldf['f'][:]
        # Include any dataset that starts with a small d
        data_list = [value for key, value in oldf['PSD'].items()
                     if key.startswith('d')]
        newf['PSD'] = np.empty((oldf['f'].value.size, len(data_list)))
        for i, val in enumerate(data_list):
            newf['PSD'][:, i] = val[:]
        oldf.close()
        newf.close()
