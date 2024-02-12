# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:17:29 2023

@author: kobe
"""

import numpy       as     np
import helpTools   as     hlp
from   helpToolsT2 import EpyG_mSE_decay


class Gaussians():
    
    def __init__(self):
        
        bla = 'bla'
    
    def add_rician_noise(self, signal, SNR):
        
        """
        Add rician noise to a real-valued signal with amplitude signal and given SNR/biais.
        If the input signal is an array, noise is scaled to first element of signal array

        returns:
        noisy signal
        """
        
        np.random.seed(0)
        
        self.noisy_signal = signal + np.random.normal(loc = 0, scale = signal[0] / SNR, size = np.shape(signal))

        return self.noisy_signal
    
    
    
    def T1_signal(self, mwf, T1min, T1max, timepoints, SNR=150, addNoise=False):
        
        """
        computes T1 decay signal from input MWF (modeled as two Gaussians)

        Input\n 
        mwf          : the true MWF\n
        T1min, T1max : minimum / maximum value of T1 relaxation times (measurements)\n
        timepoints   : sample vector (np array) at which the output signal is modeled\n
        SNR          : signal-to-noise-ratio\n

        Output       the modeled T1 decay signal
        """
        
        T1_values = np.array([100, 1000])      # centers of the Gaussians
        T1_weights = np.array([mwf, 1-mwf])    # integrals of the Gaussians
        T1_widths = np.array([20, 100])        # standard deviation of the Gaussians   
        T1_spectrum = hlp.gauss_sum(T1_values, T1_widths, T1_weights)
        
        signal = hlp.signal_from_distribution(T1_spectrum, timepoints, T1min, T1max, hlp.T1_decay_abs)

        if addNoise:
            signal = self.add_rician_noise(signal, SNR)
        
        return signal
    
    
    
    def T2star_signal(self, mwf, T2min ,T2max, timepoints, SNR=150, addNoise=False):
        
        """
        computes T2* decay signal from input MWF (modeled as two Gaussians)
    
        Input 
        mwf          : the true MWF
        T2min, T2max : minimum / maximum value of T1 relaxation times (measurements)
        timepoints   : sample vector (np array) at which the output signal is modeled
        SNR          : signal-to-noise-ratio 
    
        Output       the modeled T2* decay signal
        """
         
        T2s_values = np.array([8, 80])          # centers of the Gaussians
        T2s_weights = np.array([mwf, 1-mwf])    # integrals of the Gaussians
        T2s_widths = np.array([2, 8])           # standard deviation of the Gaussians
        T2s_spectrum = hlp.gauss_sum(T2s_values, T2s_widths, T2s_weights)
        
        self.T2s_signal = hlp.signal_from_distribution(T2s_spectrum, timepoints, T2min, T2max, hlp.T2_decay)
        
        if addNoise:
            self.T2s_signal = self.add_rician_noise(self.T2s_signal, SNR)
        
        return self.T2s_signal



    def T2_signal(self, mwf, T2min, T2max, EpyG_params, T2_grid, addNoise=False, SNR=150):
        """
        computes T2 decay signal from input MWF (modeled as two Gaussians)
    
        Input 
        mwf          : the true MWF
        T2min, T2max : minimum / maximum value of T1 relaxation times (measurements)
        SNR          : signal-to-noise-ratio 
    
        Output       the modeled T2 decay signal
        """

        T2_values = np.array([8, 80])
        T2_weights = np.array([mwf, 1-mwf])
        T2_widths = np.array([2, 8]) 
        T2_spectrum = hlp.gauss_sum(T2_values, T2_widths, T2_weights)
        
        timepoints, T2_signal = EpyG_mSE_decay(EpyG_params, T2_grid, gaussian=T2_spectrum)
        
        if addNoise:
            T2_signal = self.add_rician_noise(self.T2_signal, SNR)
        
        return T2_signal