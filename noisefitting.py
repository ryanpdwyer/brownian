"""
Noise Fitting Brownian Motion Data
Ryan Dwyer
2013-11-22

This document will implement John Marohn's Brownian motion noise 
fitting routine in Python. The file has the following stuructre.

    class BrownianMotionFitter
        This is the main object that stores, fits, and plots the data.

"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
from pint import UnitRegistry

u = UnitRegistry()
# Backend should be set correctly here.

k_B = 1.3806504e-23 * u.J / u.K

class BrownianMotionFitter(object):
    """This object fits Brownian motion position fluctuation data to
    the equation:


    estimates is a dictionary of 

    """

    def __init__(self, f, PSD, T, estimates):
        """
        f
            frequency (Hz)

        PSD
            Power spectral density of position fluctuations (nm^2 / Hz)

        estimates
            A dictionary containing initial estimates of f_c, k_c, and Q.

        """
        self.f = f
        self.PSD_raw = PSD
        self.T = T * u.K
        
        missing = [key for key in ['f_c', 'k_c', 'Q'] if key not in estimates] 
        if len(missing) == 0:
            self.estimates = estimates
        else:
            raise ValueError("The dictionary 'estimates' is missing keys:\
                                            {missing}".format(missing=missing))

    def guess_P_detector(self):
        self.P_detector0_raw = np.percentile(self.PSD, 2)

    def scale_data(self):
        self.PSD = self.PSD_raw / self.P_detector0
        self.P_detector0 = 1


    #def calc_fit(self, f_min = 1e3, f_max = 1e5):
    #    self.f_min = f_min
    #    self.f_max = f_max
    #
    #    self.guess_P_detector()
    #    self.scale_data()
    #    
    #    self._first_pass()
    #    self._second_pass()
    #    self._final_passes()

    def calc_initial_params(self):
        f_c = self.estimates['f_c'] * u.Hz
        k_c = self.estimates['k_c'] * u.N / u.m
        Q = self.estimates['Q']

        P_x0guess = (P_x0(f_c, k_c, Q, self.T) / 
                            (self.P_detector0 * u.nm ** 2 / u.Hz))
        
        self.inital_params = [P_x0guess.magnitude, f_c.magnitude, Q.magnitude]

    def _first_pass(self):
        """The first pass of the fitting protocol. We use user input guesses for
        the three cantilever parameters, and fix the thermal noise floor to a 
        low but reasonable guess. Then, we fit the remaining three parameters
        (P_x0, f_c, Q). The output of this fit is saved to 
        self.popt1, self.pcov1.
        """
        
        self.calc_initial_params()

        def Pf_fixed_P_detector(f, P_x0, f_c, Q):
            return Pf(f, P_x0, f_c, Q, self.P_detector0)

        self.popt1, self.pcov1 = curve_fit(Pf_fixed_P_detector,
                                    self.f, self.PSD,
                                    p0=self.initial_params,
                                    sigma=self.PSD * 0.1 # Make this real
                                    )


def P_x0(f_c, k_c, Q, T = 300 * u.K):
    """Used to estimate an initial value for P_x0. Input values
    using Pint for units."""
    return (2 * k_B * T / (np.pi * k_c * Q * f_c)).ito(u.nm ** 2 / u.Hz)

def Pf(f, P_x0, f_c, Q, P_detector):
    """

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
    return  (P_x0 * f_c**4 / 
                    ((f**2 - f_c**2)**2 + f**2 * f_c**2 / Q**2)
                                                             + P_detector)
