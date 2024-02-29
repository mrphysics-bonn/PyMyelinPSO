#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""

import sys
import numpy as np

from EpyG import EpyG as epyg
from EpyG import Operators as ops


def multiline_decay_create(T2_grid, weights):
    """
    Return a function representing the decay curve for localised T2-values (i.e. a(T2) is a sum of dirac delta distributions).
    T2_grid: decay parameters
    weights: weights/amplitudes of the lines
    n: length of T2_grid and weights
    returns: multiline_decay(t): decay curve given by the sum of weights[i]*exp(t/T_values[i])
    """
    def multiline_decay(t):  # the function to be returned
        sumvalue = 0
        for i in range(0, len(T2_grid)):
            sumvalue += weights[i] * np.exp(-t / T2_grid[i])

        return sumvalue

    return multiline_decay


# potential improvements to this code:
# -) create a main function to make this a module usable with as little function calls as possible
# -) automatically determine integration limits from mu_values and sigma_values (i.e. min(mu_values-5*sigma_values))


def multiline_signal(T2_line_spectrum, weights_true, timepoints):
    """
    Generate a signal from a multiline decay function with the option to add noise.
    Return noisy signal and noiseless signal.
    """
    # generating function describing the 'true' decay and /giphy vector noisy_decay holding the true decay 'measurements'
    true_decay = multiline_decay_create(T2_line_spectrum, weights_true)(timepoints)

    return true_decay  # if noise is 0, noisy decay is equal to noiseless decay...



class EpyG_Paramset:
    """A class that holds parameters needed for EpyG analysis to make calling functions a bit more clear."""

    def __init__(self, T1, flipangle, T_E, ETL):
        """Initialise EpyG Parameterset object."""
        self.T1 = T1
        self.flipangle = flipangle
        self.T_E = T_E
        self.ETL = ETL

def epyg_amplitudes(T1, T2, beta, ESP, ETL):
    '''
    Using epg algorithm of Daniel Brenner.
    Calculates the amplitude of coherent pathway for a spin echo sequence.
    Output is an array, which contains the amplitude of every echo in ETL
    
    T1: T1 of sample
    T2: T2 of sample
    beta: Refocussing flip angle
    ESP: Echo spacing
    ETL: Echo train length'''
   
    #90deg excitation puls
    T_ex = ops.Transform(alpha=np.deg2rad(beta/2), phi=0.0, name="Excitation")
    #180deg refocussierungspuls
    T_ref = ops.Transform(alpha=np.deg2rad(beta), phi=np.deg2rad(90.0), name="Excitation")
    #Shift operator
    S = ops.Shift(shifts=1, autogrow=True, name="Dephase")
    #relaxation operator
    E = ops.Epsilon(TR_over_T1 = ESP/(2*T1), TR_over_T2 = ESP/(2*T2), name="Relaxation")
    #observer operator
    O = ops.Observer(f_states=(0,), z_states=(0,), name="ADC")
    
    # Composite operator fuehrt nacheinander aus: shift + relaxation for TE / 2 til refocussing puls,
    # shift + relaxation for TE/2 til observation of echo amplitude
    c = ops.CompositeOperator(O, E, S, T_ref, E, S, name = "mSE")
    
    # produziere equilibrium longitudinale magnetisierung
    se_graph = epyg.EpyG(initial_size=256, m0=1.0)
    
    #excitation
    T_ex * se_graph
    for i in np.arange(0, ETL, 1):
        c*se_graph
        
    return np.abs(O.get_f(0))


def EpyG_mSE_decay(EpyG_params, T2_grid, T1_grid=None, **kwargs):
    """
    Create decay corresponding to a multispinecho sequence from the Input parameters.
    Using Daniel Brenner's EpyG Algorithm.
    Input Parameters: EpyG_params contains:
    T1: T1 of sample
    T2_stepnumber: number of T2 gridpoints to be analysed
    flipangle: sequence flip angle
    T_E: Sequence echo time (time difference between echoes)
    ETL: Sequence echo train length
    Kwargs:
    mode: specifies spectrum from which the decay is generated
    gaussian: gaussian class in case of gaussian input spectrum.
    T2_lines: line spectrum decay parameters for multiline mode
    weights = weights of spectral lines in multiline mode
    """
    if not T1_grid:
        T1 = EpyG_params.T1
    flipangle = EpyG_params.flipangle
    T_E = EpyG_params.T_E
    ETL = EpyG_params.ETL

    mode = kwargs.get('mode', None)

    decay_timepoints = np.zeros(ETL)
    for i in range(ETL):
        decay_timepoints[i] = i * T_E + T_E
    decay_amplitudes = np.zeros(ETL)

    if mode == 'gaussian' or not mode:  # default mode is gaussian
        gaussian = kwargs.get('gaussian', None)
        assert gaussian, 'No gaussian given to create decay from'
        # discretising gaussian in a fitting way
        weights = gaussian.fast_convergent_weights(T2_grid)

    elif mode == 'multiline':
        weights = kwargs.get('weights', None)
        T2_lines = kwargs.get('T2_lines', None)
#        assert weights and T2_lines, 'weights and/or T2 lines not specified'
        T2_grid = T2_lines  # don't need a grid in this mode, just the lines given
        # the +50 is arbitrary just so the spectrum has some room to the left and the right

    else:
        print('unknown mode specification "', mode, '" cannot create EpyG decay.')
        sys.exit()

    for i in range(len(T2_grid)):
        decay_amplitudes += weights[i] * epyg_amplitudes(T1, T2_grid[i],
                                                                         flipangle, T_E, ETL)

    return decay_timepoints, decay_amplitudes


def EpyG_mSE_system_matrix(EpyG_params, T2_grid):
    """
    Create multi-spi-echo system matrix for NNLS
    Input:
        EpyG_params:    EpyG_Paramset(T1, flipangle, TE, ETL)
        T2_grid
    """
    T1 = EpyG_params.T1
    flipangle = EpyG_params.flipangle
    T_E = EpyG_params.T_E
    ETL = EpyG_params.ETL

    timepoints = np.zeros(ETL)
    A = np.zeros((ETL,len(T2_grid)))

    for i in range(len(T2_grid)):
        A[:,i] = epyg_amplitudes(T1, T2_grid[i],flipangle, T_E, ETL)

    return A
