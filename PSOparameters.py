# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 13:34:00 2023

@author: kobe
"""

import numpy as np

class Parameters():
    
    def __init__(self, noPart, noIter):
        
        self.Inversion = self.Inversion()
        self.PSO       = self.PSO(noPart, noIter)
        self.T1        = self.T1()
        self.T2        = self.T2()
        self.T2S       = self.T2S()    
    
    class Inversion():
        
        def __init__(self):
            
            # General parameters
            self.datSpaceSyn    = 20    # step size for the plot of the syn/calc Data
            self.datSpaceT1     = 20    # step size for the plot of the syn/meas Data
            self.datSpaceT2     = 24    # step size for the plot of the syn/meas Data
            self.datSpaceT2S    = 32    # step size for the plot of the syn/meas Data
            self.modSpace       = 1000  # step size for integration
            self.SNR            = 100   # relatively added noice 1/SNR
            
            # T1 parameters for generation of observed data from MWF
            self.T1_TR          = 4      # repetition time
            self.T1_alpha       = 4.0    # ???
            self.T1_TD          = 1000.0 # ???
            self.T1_IE          = 0.95   # ???
            self.T1min          = 1      # integration lower boarder
            self.T1max          = 2000   # integration upper boarder
            self.T1_timepoints  = np.linspace(1, 5000, self.datSpaceT1)
            
            # T2** parameters for generation of observed data from MWF
            self.T2Smin         = 1     # integration lower boarder
            self.T2Smax         = 200   # integration upper boarder
            self.T2S_timepoints = np.linspace(1, 150, self.datSpaceT2S)
            
            # T2 parameters for generation of observed data from MWF
            # --> multi-Spin-Echo (mSE) sequence parameters
            # --> original TE: 5 / flipangle: 120
            # --> Test#1: 5/180  und  Test#2: 6/180
            # te: echo time
            # tr: repetition time
            # T2_T1:
            self.T2min            = 1
            self.T2max            = 200
            self.T2_TE            = 6.6
            self.T2_TR            = 900
            self.T2_alpha         = 70
            self.T2_beta          = 180
            self.T2_flipangle     = 120
            self.T2_ETL           = self.datSpaceT2
            self.T2_T1            = 1000  
            self.T2epg_timepoints = np.linspace(self.T2_TE, 
                                                self.T2_ETL*self.T2_TE, 
                                                self.T2_ETL)
            
    class PSO():
        
        def __init__(self, noPart, noIter):
            
            self.noPart = noPart  # particle number the particle swarm contains
            self.w      = 0.7298  # inertia weight factor
            self.c1     = 1.4962  # social weight factor
            self.c2     = 1.4962  # cognitive weight factor
            self.noIter = noIter  # iteration numbers per execution
            
    class T1():
        
        def __init__(self):
            
            self.GAUSS = self.GAUSS()
            self.DIRAC = self.DIRAC()
        
        class GAUSS():
            
            def __init__(self):         
                self.noPara = 6            # model vector size
                self.m1     = (50,   300)  # center of the gaussian, 1st peak  
                self.m1_sig = (0.1,  50)   # standard deviation of m1
                self.m2     = (700,  1300) # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1,  150)  # standard deviation of m2
                self.int2   = (0.1,  5)    # area under the curve of m2 gaussian
                self.MWF    = (0,    0.85) # typical intervall of MWF in a MRT

        class DIRAC():
            
            def __init__(self):         
                self.noPara = 9            # model vector size
                self.m1     = (50,   300)  # center of the gaussian, 1st peak  
                self.m1_sig = (0.1,  50)   # standard deviation of m1
                self.m2     = (700,  1300) # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1,  150)  # standard deviation of m2
                self.m3     = (1300, 2000) # center of the gaussian, 3rd peak
                self.m3_sig = (0.1,  0.1)  # standard deviation of m2
                self.int2   = (0.1,  5)    # area under the curve of m2 gaussian
                self.int3   = (0.1,  5)    # area under the curve of m3 gaussian
                self.MWF    = (0,    0.85) # typical intervall of MWF in a MRT
                
    class T2():
        
        def __init__(self):
            
            self.GAUSS = self.GAUSS()
            self.DIRAC = self.DIRAC()
        
        class GAUSS():
            
            def __init__(self):
                self.noPara = 6           # model vector size
                self.m1     = (5,   35)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                self.m2     = (60,  100)  # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 5)    # standard deviation of m2
                self.int2   = (0,   5)    # area under the curve of m2 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT
            
        class DIRAC():
            
            def __init__(self):
                self.noPara = 9           # model vector size
                self.m1     = (5,   35)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                self.m2     = (60,  90)   # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 5)    # standard deviation of m2
                self.m3     = (80,  130)  # center of the gaussian, 3rd peak
                self.m3_sig = (5,   40)   # standard deviation of m2
                self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                self.int3   = (0.1, 10)   # area under the curve of m3 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT
    
    class T2S():
        
        def __init__(self):
            
            self.GAUSS = self.GAUSS()
            self.DIRAC = self.DIRAC()
        
        class GAUSS():
            
            def __init__(self):                
                self.noPara = 6           # model vector size
                self.m1     = (5,   35)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                self.m2     = (51,  130)  # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 5)    # standard deviation of m2
                self.int2   = (0,   5)    # area under the curve of m2 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT   

        class DIRAC():
            
            def __init__(self):
                self.noPara = 9           # model vector size
                self.m1     = (5,   35)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                self.m2     = (40,  80)   # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 5)    # standard deviation of m2
                self.m3     = (60,  120)  # center of the gaussian, 3rd peak
                self.m3_sig = (5,   40)   # standard deviation of m2
                self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                self.int3   = (0.1, 10)   # area under the curve of m3 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT        
