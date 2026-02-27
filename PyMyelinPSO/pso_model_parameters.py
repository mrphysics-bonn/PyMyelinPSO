# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# Author:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of the PyMRI_PSO software.
# See the LICENSE file in the project root for full license information.

"""
Parameter definitions for particle swarm optimization (PSO) applied to in vivo MRI data.
    
    InversionParams - signal model parameters (mwf_analysis)
    PSOParams       - literature-based weight factors
    T1/T2/T2SParams - parameter intervals for the model vector
"""

import numpy as np

class Parameters():
    
    def __init__(self):
        
        self.Inv = self.InversionParams()
        self.PSO = self.PSOParams()
        self.T1  = self.T1Params()
        self.T2  = self.T2Params()
        self.T2S = self.T2SParams()    
    
    class InversionParams():
        
        def __init__(self):
            
            # General parameters
            self.n_echoes_T1  = 20    # number of echo times for T1
            self.n_echoes_T2  = 24    # number of echo times for T2
            self.n_echoes_T2S = 32    # number of echo times for T2S (mag & phs)
            self.mod_space    = 1000  # number of samples in the model space
            
            # T1 model parameters
            self.T1_TR    = 4      # repetition time [ms]
            self.T1_alpha = 4.0    # excitation flip angle [deg]
            self.T1_TD    = 1000   # variable inversion-recovery delay [ms]
            self.T1_IE    = 0.95   # inversion efficiency
            self.T1_min   = 1      # lower bound of T1 search space [ms]
            self.T1_max   = 2000   # upper bound of T1 search space [ms]
            
            # T2 model parameters
            self.T2_min   = 1      # lower bound of T2 search space [ms]    
            self.T2_max   = 200    # upper bound of T2 search space [ms]
            self.T2_TE    = 6.6    # echo time / step size [ms]
            self.T2_TR    = 2000   # repetition time [ms]
            self.T2_alpha = 90     # excitation flip angle [deg]
            self.T2_beta  = 160    # refocusing flip angle [deg]
            self.T2_ETL   = self.n_echoes_T2  # echo train length
            self.T2_T1    = 1000   # T₁ relaxation time used in the model [ms]             
                        
            # T2S model parameters
            self.T2S_min  = 1      # lower bound of T2S search space [ms]
            self.T2S_max  = 200    # upper bound of T2S search space [ms]
            
    class PSOParams():
        
        def __init__(self):
            
            self.w      = 0.7298  # inertia weight factor
            self.c1     = 1.4962  # social weight factor
            self.c2     = 1.4962  # cognitive weight factor
            
    class T1Params():
        
        def __init__(self):
            
            self.TwoComponentParams   = self.TwoComponentParams()
            self.ThreeComponentParams = self.ThreeComponentParams()
        
        class TwoComponentParams():
            
            def __init__(self):         
                self.m1      = (50,  300)  # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.1, 50)   # std (σ) / 1st Gaussian width
                self.m2      = (700, 1300) # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.1, 150)  # std (σ) / 2nd Gaussian width
                self.int2    = (0.1, 5)    # area under the 2nd Gaussian (integral) 
                self.MWF     = (0,   0.85) # myelin water fraction (integral)

        class ThreeComponentParams():
            
            def __init__(self):         
                self.m1      = (50,   300)  # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.1,  10)   # std (σ) / 1st Gaussian width
                self.m2      = (700,  1300) # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.1,  10)   # std (σ) / 2nd Gaussian width
                self.m3      = (1300, 2000) # mean (μ) / 3rd Gaussian center
                self.m3_sig  = (0.1,  0.1)  # std (σ) / 3rd Gaussian width
                self.int2    = (0.1,  5)    # area under the 2nd Gaussian (integral) 
                self.int3    = (0.1,  5)    # area under the 3rd Gaussian (integral) 
                self.MWF     = (0,    0.85) # myelin water fraction (integral)
                
    class T2Params():
        
        def __init__(self):
            
            self.TwoComponentParams   = self.TwoComponentParams()
            self.ThreeComponentParams = self.ThreeComponentParams()
        
        class TwoComponentParams():
            
            # def __init__(self):
                # self.m1      = (10,  25)   # mean (μ) / 1st Gaussian center
                # self.m1_sig  = (0.1, 0.1)  # std (σ) / 1st Gaussian width
                # self.m2      = (60,  100)  # mean (μ) / 2nd Gaussian center
                # self.m2_sig  = (0.1, 0.5)  # std (σ) / 2nd Gaussian width
                # self.int2    = (0.1, 10)   # area under the 2nd Gaussian (integral) 
                # self.MWF     = (0,   0.85) # myelin water fraction (integral)
            
            # # for atlas
            def __init__(self):
                self.m1      = (10,  25)   # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.5, 0.5)  # std (σ) / 1st Gaussian width
                self.m2      = (60,  100)  # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.5, 0.5)  # std (σ) / 2nd Gaussian width
                self.int2    = (0.1, 10)   # area under the 2nd Gaussian (integral) 
                self.MWF     = (0,   0.85) # myelin water fraction (integral)
        
        class ThreeComponentParams():
            
            def __init__(self):
                self.m1      = (5,   45)   # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.1, 10) # std (σ) / 1st Gaussian width
                self.m2      = (60,  85)   # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.1, 10) # std (σ) / 2nd Gaussian width
                self.m3      = (90,  110)  # mean (μ) / 3rd Gaussian center
                self.m3_sig  = (0.1, 10) # std (σ) / 3rd Gaussian width
                self.int2    = (0.1, 5)    # area under the 2nd Gaussian (integral) 
                self.int3    = (0.1, 5)    # area under the 3rd Gaussian (integral) 
                self.MWF     = (0,   0.85) # myelin water fraction (integral)

    class T2SParams():
        
        def __init__(self):
            self.TwoComponentParams   = self.TwoComponentParams()
            self.ThreeComponentParams = self.ThreeComponentParams()
        
        class TwoComponentParams():
            
            def __init__(self):  
                self.m1      = (5,   25)   # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.1, 0.1)  # std (σ) / 1st Gaussian width
                self.m2      = (45,  75)   # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.1, 0.5)  # std (σ) / 2nd Gaussian width
                self.int2    = (0.1, 5)    # area under the 2nd Gaussian (integral) 
                self.MWF     = (0,   0.85) # myelin water fraction (integral)
                self.MW_f    = (-25, 25)   # frequency shift MW component (Hz)
                self.FW_f    = (-25, 25)   # frequency shift FW component (Hz)
                self.phi     = (0, 2*np.pi) # global phase shift (rad)
                
                # # for atlas
                # self.m1      = (5,   25)    # mean (μ) / 1st Gaussian center
                # self.m1_sig  = (0.1, 0.1)   # std (σ) / 1st Gaussian width
                # self.m2      = (45,  65)    # mean (μ) / 2nd Gaussian center (60, 90)
                # self.m2_sig  = (0.1, 0.5)   # std (σ) / 2nd Gaussian width
                # self.int2    = (0.1, 5)     # area under the 2nd Gaussian (integral) 
                # self.MWF     = (0,   0.85)  # myelin water fraction (integral)
                # self.MW_f    = (-25, 25)    # frequency shift MW component (Hz)
                # self.FW_f    = (-25, 25)    # frequency shift FW component (Hz)
                # self.phi     = (0, 2*np.pi) # global phase shift (rad)

        class ThreeComponentParams():
            
            def __init__(self):
                self.m1      = (5,   25)    # mean (μ) / 1st Gaussian center
                self.m1_sig  = (0.1, 0.10)  # std (σ) / 1st Gaussian width
                self.m2      = (50,  80)    # mean (μ) / 2nd Gaussian center
                self.m2_sig  = (0.1, 0.10)  # std (σ) / 2nd Gaussian width
                self.m3      = (90,  130)   # mean (μ) / 3rd Gaussian center
                self.m3_sig  = (0.1, 0.10)  # std (σ) / 3rd Gaussian width
                self.int2    = (0.1, 5)     # area under the 2nd Gaussian (integral) 
                self.int3    = (0.1, 5)     # area under the 3rd Gaussian (integral) 
                self.MWF     = (0,   0.85)  # myelin water fraction (integral)
                self.MW_f    = (-25, 25)    # frequency shift MW component (Hz)
                self.EW_f    = (-25, 25)    # frequency shift EW component (Hz)
                self.AW_f    = (-25, 25)    # frequency shift AW component (Hz)                
                self.phi     = (0, 2*np.pi) # global phase shift (rad)           