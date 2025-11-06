# -*- coding: utf-8 -*-
"""
Initial parameters for applying particle swarm optimizing (PSO) on invivo MRI data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)
"""

import numpy as np

class Parameters():
    
    def __init__(self):
        
        self.Inversion = self.Inversion()
        self.PSO       = self.PSO()
        self.T1        = self.T1()
        self.T2        = self.T2()
        self.T2S       = self.T2S()    
    
    class Inversion():
        
        def __init__(self):
            
            # General parameters
            self.datSpaceSyn    = 20    # step size for the plot of the syn/calc Data
            self.datSpaceT1     = 20    # step size for the plot of the syn/meas Data
            self.datSpaceT2     = 24    # step size for the plot of the syn/meas Data # normally 24 --> why sego uses 60 ??? 
            self.datSpaceT2S    = 32    # step size for the plot of the syn/meas Data # normally 32 --> why sego uses 60 ??? 
            self.modSpace       = 1000  # step size for integration
            self.SNR            = 100   # relatively added noice 1/SNR
            
            # T1 parameters for generation of observed data from MWF
            self.T1_TR          = 4      # repetition time
            self.T1_alpha       = 4.0    # ???
            self.T1_TD          = 1000.0 # ???
            self.T1_IE          = 0.95   # ???
            self.T1min          = 1      # integration lower border
            self.T1max          = 2000   # integration upper border
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
            # T2_T1: integration room size
            self.T2min            = 1                        # 1    | sego: 1             
            self.T2max            = 200                      # 200  | sego: 200
            self.T2_TE            = 6.6                      # 6.6  | sego: 6.0
            self.T2_TR            = 2000                     # 900  | sego: 2000
            self.T2_alpha         = 90                       # 70   | sego: 90
            self.T2_beta          = 160                      # 180  | sego: 160
            self.T2_ETL           = self.datSpaceT2          # 24   | sego: 24
            self.T2_T1            = 1000                     # 1000 | sego: 1000
            self.T2epg_timepoints = np.linspace(self.T2_TE, 
                                                self.T2_ETL*self.T2_TE, 
                                                self.T2_ETL)
            
    class PSO():
        
        def __init__(self):
            
            self.w      = 0.7298  # inertia weight factor
            self.c1     = 1.4962  # social weight factor
            self.c2     = 1.4962  # cognitive weight factor
            
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
                self.m1_sig = (0.1,  10)  # standard deviation of m1
                self.m2     = (700,  1300) # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1,  10)  # standard deviation of m2
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
                self.m1     = (10,  25)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                self.m2     = (60,  100)   # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 0.5)  # standard deviation of m2
                self.int2   = (0.1, 10)   # area under the curve of m2 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT   
            
            # def __init__(self):
            #     self.noPara = 6           # model vector size
            #     self.m1     = (10,  25)   # center of the gaussian, 1st peak 
            #     self.m1_sig = (0.1, 0.1)  # standard deviation of m1
            #     self.m2     = (60,  85)   # center of the gaussian, 2nd peak 
            #     self.m2_sig = (0.1, 0.5)  # standard deviation of m2
            #     self.int2   = (0.1, 10)   # area under the curve of m2 gaussian
            #     self.MWF    = (0,   0.85) # typical intervall of MWF in a MRT   
            
        # class DIRAC():
            
        #     def __init__(self):
        #         self.noPara = 9            # model vector size
        #         self.m1     = (5,   25)    # center of the gaussian, 1st peak 
        #         self.m1_sig = (0.1, 10)  # standard deviation of m1
        #         self.m2     = (45,  85)    # center of the gaussian, 2nd peak 
        #         self.m2_sig = (0.1, 10)  # standard deviation of m2
        #         self.m3     = (85,  150)   # center of the gaussian, 3rd peak
        #         self.m3_sig = (0.1, 10)  # standard deviation of m2
        #         self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
        #         self.int3   = (0.1, 5)    # area under the curve of m3 gaussian
        #         self.MWF    = (0,   0.85)  # typical intervall of MWF in a MRI
        
        class DIRAC():
            
            def __init__(self):
                self.noPara = 9           # model vector size
                self.m1     = (5,   45)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.10)   # standard deviation of m1
                self.m2     = (60,  85)   # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 0.10)   # standard deviation of m2
                self.m3     = (80,  110)  # center of the gaussian, 3rd peak
                self.m3_sig = (0.1, 0.10)   # standard deviation of m2
                self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                self.int3   = (0.1, 5)    # area under the curve of m3 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRI
                
                # self.noPara = 9           # model vector size
                # self.m1     = (10,  25)   # center of the gaussian, 1st peak 
                # self.m1_sig = (0.1, 0.10)   # standard deviation of m1
                # self.m2     = (60,  85)   # center of the gaussian, 2nd peak 
                # self.m2_sig = (0.1, 0.50)   # standard deviation of m2
                # self.m3     = (80,  110)  # center of the gaussian, 3rd peak
                # self.m3_sig = (0.1, 0.50)   # standard deviation of m2
                # self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                # self.int3   = (0.1, 5)    # area under the curve of m3 gaussian
                # self.MWF    = (0,   0.85) #

    class T2S():
        
        def __init__(self):
            
            self.GAUSS = self.GAUSS()
            self.DIRAC = self.DIRAC()
        
        class GAUSS():
            
            def __init__(self):  
                # self.noPara = (6,   9)    # model vector size
                # self.m1     = (10,  25)    # center of the gaussian, 1st peak 
                # self.m1_sig = (0.1, 0.1)  # standard deviation of m1
                # self.m2     = (60,  90)   # center of the gaussian, 2nd peak 
                # self.m2_sig = (0.1, 5)    # standard deviation of m2
                # self.int2   = (0.1, 5)   # area under the curve of m2 gaussian
                # self.MWF    = (0,   0.85) # typical intervall of MWF in a MRI
                # self.MW_f   = (-75, 75)   # frequency shift MW component (Hz)
                # self.FW_f   = (-75, 75)   # frequency shift FW component (Hz)
                # self.phi    = (-np.pi, np.pi)   # global phase shift
                
                # for atlas
                self.noPara = (6,   9)    # model vector size
                self.m1     = (5,   20)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.1)    # standard deviation of m1
                self.m2     = (45,  65)  # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 0.1)   # standard deviation of m2
                self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRI
                self.MW_f   = (-25, 25)   # frequency shift MW component (Hz)
                self.FW_f   = (-25, 25)   # frequency shift FW component (Hz)
                self.phi    = (0, 2*np.pi)   # global phase shift
                # self.phi    = (-15, 15)   # global phase shift

        class DIRAC():
            
            def __init__(self):
                self.noPara = (9,   13)   # model vector size
                self.m1     = (5,   25)   # center of the gaussian, 1st peak 
                self.m1_sig = (0.1, 0.10)  # standard deviation of m1
                self.m2     = (50,  80)   # center of the gaussian, 2nd peak 
                self.m2_sig = (0.1, 0.10)  # standard deviation of m2
                self.m3     = (90,  130)  # center of the gaussian, 3rd peak
                self.m3_sig = (0.1, 0.10)  # standard deviation of m2
                self.int2   = (0.1, 5)    # area under the curve of m2 gaussian
                self.int3   = (0.1, 5)    # area under the curve of m3 gaussian
                self.MWF    = (0,   0.85) # typical intervall of MWF in a MRI
                self.MW_f   = (-75, 75)   # frequency shift MW component (Hz)
                self.EW_f   = (-25, 25)   # frequency shift EW component (Hz)
                self.AW_f   = (-25, 25)   # frequency shift AW component (Hz)                
                self.phi    = (-15, 15)   # global phase shift                