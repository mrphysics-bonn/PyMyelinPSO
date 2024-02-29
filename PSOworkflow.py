# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:59:02 2023

@author: kobe
"""

import time, sys, os, copy, json, matplotlib
import numpy             as     np
import matplotlib.pyplot as     plt
from   sklearn.cluster   import DBSCAN
from   scipy.ndimage     import gaussian_filter, label, generate_binary_structure

import helpTools         as     hlp
from   helpToolsT2       import EpyG_Paramset, EpyG_mSE_system_matrix
from   kernels           import T1_decay_abs, T2_decay, system_matrix_from_kernel
from   parameters        import Parameters   as PM
from   inversion         import Gaussians    as Integrals

# initialisieren oder vererben !!!
# gaus = Integrals()

class ParticleSwarmOptimizer(PM, Integrals):
    
    def __init__(self,
                 noPart,
                 noIter,
                 noPSOIter,
                 signal_input,
                 signal_type,
                 *args,
                 **kwargs):
        
        # Inherit parameters and methods from:
        # (a) class containing parameter for
        #     (a.a) generating obsData and synData of T1, T2, T2*
        #     (a.b) performing PSO algorithm and cycles
        # (b) forward solver class inversion.Gaussian
        #     (b.a) methods for generating obsData T1, T2, T2* via integration
        super().__init__(noPart,noIter)

        # possible keyword arguments, e.g. randseed
        self.randSeedValue = kwargs.get('randSeed', (0, False))[0]
        self.randSeedBool  = kwargs.get('randSeed', (0, False))[1]
        self.lpNorm        = kwargs.get('lpNorm',   'L2')
        self.noPeaks       = kwargs.get('noPeaks',  'GAUSS')
        
        if self.randSeedBool == True:
            np.random.seed(self.randSeedValue)

        # General parameters
        self.invT1     = signal_input[1][0]
        self.invT2     = signal_input[1][1]
        self.invT2S    = signal_input[1][2]
        self.addNoise  = signal_input[1][3]        
        self.signType  = signal_type

        self.signal_cc = sum(signal_input[1][:3])             # RELEVANT FOR JI
        self.singleInv = signal_input[1][4]                   # RELEVANT FOR JI
        self.jointInv  = signal_input[1][5]                   # RELEVANT FOR JI
        self.optionInv = signal_input[1][6]                   # RELEVANT FOR JI
        self.dataType  = signal_input[1][7]

        self.noParam   = self.T2.GAUSS.noPara
        self.noIter    = self.PSO.noIter
        self.noPart    = self.PSO.noPart    
        self.noPSOIter = noPSOIter
        self.SNR       = self.Inversion.SNR
        
        if self.jointInv==True:
            self.sigJoint   = f'{self.signType[0]}{self.signType[1]}'
        else:
            self.sigJoint   = 'None'
        
        # Inversion/Integration-parameters for T1, T2 and T2**     
        # MxN means in the context of numpy M rows and N columns
        self.sysMatrix = {'T1':[],'T2':[],'T2S':[]}
        self.sysGrid   = {'T1':[],'T2':[],'T2S':[]}
        self.m1        = {'T1':[],'T2':[],'T2S':[]}
        self.m1_sig    = {'T1':[],'T2':[],'T2S':[]}
        self.m2        = {'T1':[],'T2':[],'T2S':[]}
        self.m2_sig    = {'T1':[],'T2':[],'T2S':[]}
        self.m3        = {'T1':[],'T2':[],'T2S':[]}
        self.m3_sig    = {'T1':[],'T2':[],'T2S':[]}
        self.int2      = {'T1':[],'T2':[],'T2S':[]}
        self.int3      = {'T1':[],'T2':[],'T2S':[]}
        self.MWF       = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        
        # PSO parameters
        self.modV1     = {'T1':[],'T2':[],'T2S':[]}
        self.modV2     = {'T1':[],'T2':[],'T2S':[]}
        self.vel       = {'T1':[],'T2':[],'T2S':[]}
        self.fit       = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        self.synDat    = {'T1':[],'T2':[],'T2S':[]}
        self.noSteps   = {'T1':  self.Inversion.datSpaceT1,
                          'T2':  self.Inversion.datSpaceT2,
                          'T2S': self.Inversion.datSpaceT2S}
 
        # Filling dictionaries with values: should be implemented as function
        for signal in self.signType:
            
            m1     = getattr(getattr(self, signal), self.noPeaks).m1
            m1_sig = getattr(getattr(self, signal), self.noPeaks).m1_sig
            m2     = getattr(getattr(self, signal), self.noPeaks).m2
            m2_sig = getattr(getattr(self, signal), self.noPeaks).m2_sig
            int2   = getattr(getattr(self, signal), self.noPeaks).int2
            MWF    = getattr(getattr(self, signal), self.noPeaks).MWF

            self.m1[signal]     = np.random.uniform(m1[0], m1[1], self.noPart)
            self.m1_sig[signal] = np.random.uniform(m1_sig[0], m1_sig[1], self.noPart)
            self.m2[signal]     = np.random.uniform(m2[0], m2[1], self.noPart)
            self.m2_sig[signal] = np.random.uniform(m2_sig[0], m2_sig[1], self.noPart)
            self.int2[signal]   = np.random.uniform(int2[0], int2[1], self.noPart)
            self.MWF[signal]    = np.random.uniform(MWF[0], MWF[1], self.noPart)

            # # modelvector-matrix in the shape 3xMxN --> 3 signals, M particles, N parameters
            # #    mod = [[[mod11][mod12][.....][mod1N]]
            # #           [[mod21][mod12][.....][mod1N]]
            # #           [[.....][.....][.....][.....]]
            # #           [[modM1][modM2][.....][modMN]]]
            self.modV1[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
                                                   self.m2[signal],   self.m2_sig[signal], 
                                                   self.int2[signal], self.MWF[signal]))
            
            if self.noPeaks == 'DIRAC':
                m3     = getattr(getattr(self, signal), self.noPeaks).m3
                m3_sig = getattr(getattr(self, signal), self.noPeaks).m3_sig
                int3   = getattr(getattr(self, signal), self.noPeaks).int3

                self.noParam        = self.T2.DIRAC.noPara
                self.m3[signal]     = np.random.uniform(m3[0], m3[1], self.noPart)
                self.m3_sig[signal] = np.random.uniform(m3_sig[0], m3_sig[1], self.noPart)
                self.int3[signal]   = np.random.uniform(int3[0], int3[1], self.noPart)

                self.modV1[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
                                                       self.m2[signal],   self.m2_sig[signal],
                                                       self.m3[signal],   self.m3_sig[signal],
                                                       self.int2[signal], self.int3[signal],
                                                       self.MWF[signal]))

            # ergibt ein 3D array mit shape (3,M,N) und Werten im Intervall [-0.3,0.3]     
            self.mod             = copy.deepcopy(self.modV1)
            self.vel[signal]     = np.random.uniform(-0.3, 0.3, (self.noPart, self.noParam))*self.mod[signal]      
            self.fit[signal]     = np.zeros(self.noPart)
            self.synDat[signal]  = np.zeros((self.noPart, self.noSteps[signal]))
        
        # Continue Filling dictionaries with values if joint inversion was chosen
        if self.jointInv==True:
            self.fit[self.sigJoint] = np.zeros(self.noPart)
            self.MWF[self.sigJoint] = np.zeros(self.noPart)
            
        # parameter settings for a best local fit        
        self.bestMWF    = {key: np.full_like(self.MWF[key],    np.inf) for key in self.MWF.keys()}
        self.bestFit    = {key: np.full_like(self.fit[key],    np.inf) for key in self.fit.keys()}
        self.bestMod    = {key: np.full_like(self.mod[key],    np.inf) for key in self.mod.keys()}
        self.bestSynDat = {key: np.full_like(self.synDat[key], np.inf) for key in self.synDat.keys()}
        
        # parameter settings for a best global fit              
        self.globFit    = {'T1':np.inf,'T2':np.inf,'T2S':np.inf,self.sigJoint:np.inf}
        self.globMod    = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        self.globSynDat = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        self.globIndex  = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}

# ###############################################################################        
#     def buildModelVector(self):
        
    
###############################################################################
    def getObsDataFromMeas(self, **kwargs):
        
        '''    
        Creates a numpy array obsData from measured MRI input data.

        Four signal types are possible to invert:
            T1 == obsData[0], T1 == obsData[1], 
            T2S == obsData[2], joint == obsData[3] --> if jointInv==True
            

        Input parameters
            path: measurement file

        Returns: 
            observed MRI relaxation data (type: numpy.ndarray)
        
        '''

        yy, xx, noSlice = kwargs.get('position', (0,0,0))
        
        pathSig         = kwargs.get('pathSignal', None)
        pathT1          = kwargs.get('pathT1', None)
        pathT2          = kwargs.get('pathT2', None)
        pathT2S         = kwargs.get('pathT2S', None)
        
        if self.singleInv == True:
            
            obsData = {'T1':[],'T2':[],'T2S':[]}
            rawData = pathSig.get_fdata()
            
            if self.invT1==True:
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T1']  = datHelp[yy,xx]
                obsData['T1']  = obsData['T1']/np.max(obsData['T1'])
            if self.invT2==True:
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T2']  = datHelp[yy,xx]
                obsData['T2']  = obsData['T2']/np.max(obsData['T2'])
            if self.invT2S==True:
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T2S'] = datHelp[yy,xx]
                obsData['T2S'] = obsData['T2S']/np.max(obsData['T2S'])

        if self.jointInv == True:
            
            obsData = {'T1':[],'T2':[],'T2S':[]}
            
            if self.invT1==True:
                rawData        = pathT1.get_fdata()
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T1']  = datHelp[yy,xx]
                obsData['T1']  = obsData['T1']/np.max(obsData['T1'])
            if self.invT2==True:
                rawData        = pathT2.get_fdata()
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T2']  = datHelp[yy,xx]
                obsData['T2']  = obsData['T2']/np.max(obsData['T2'])
            if self.invT2S==True:
                rawData        = pathT2S.get_fdata()
                datHelp        = gaussian_filter(rawData[:,:,noSlice], (1.0, 1.0, 0))
                obsData['T2S'] = datHelp[yy,xx]
                obsData['T2S'] = obsData['T2S']/np.max(obsData['T2S'])

        return obsData

###############################################################################
        
    def createObsData_slow(self, MWF: float, addNoise=False):
        
        '''    
        Creates a MRI decay signal from a Mean Water Fraction (MWF) value.

        Four signal types are possible to invert:
            T1 == obsData['T1'], T1 == obsData['T2'], T2S == obsData['T2S']
            
        The method is based on iterative integration over two gaussian curves (slow).
        
        Input parameters
        ----------
        MWF: mean water fraction value (type: float)
        
        addNoise: True or False (type: boolean)

        Returns: observed MRI relaxation data (type: numpy.ndarray)
        
        '''
        
        obsData = {'T1':[],'T2':[],'T2S':[]}
        
        if self.invT1==True:
        
            obsData['T1']    = self.T1_signal(MWF,self.Inversion.T1min,
                                              self.Inversion.T1max,self.Inversion.T1_timepoints)
        
        if self.invT2==True:
            
            self.EpyG_params = EpyG_Paramset(self.Inversion.T2_T1,self.Inversion.T2_flipangle, 
                                             self.Inversion.T2_TE,self.Inversion.T2_ETL)
            
            grid             = hlp.make_grid(self.Inversion.T2min, self.Inversion.T2max, 
                                             self.Inversion.modSpace, mode='lin')
            
            obsData['T2']    = self.T2_signal(MWF,self.Inversion.T2min,self.Inversion.T2max, 
                                              self.EpyG_params, grid)
        
        if self.invT2S==True:
            
            obsData['T2S']   = self.T2S_signal(MWF,self.Inversion.T2Smin,
                                               self.Inversion.T2Smax,self.Inversion.T2S_timepoints)
            
        if addNoise:
            if self.invT1==True:
                obsData['T1']  = obsData['T1'] + \
                np.random.normal(loc=0,scale=obsData['T1'][0]/self.SNR,size=np.shape(obsData['T1']))
            if self.invT2==True:
                obsData['T2']  = obsData['T2'] + \
                np.random.normal(loc=0,scale=obsData['T2'][0]/self.SNR,size=np.shape(obsData['T2']))
            if self.invT2S==True:
                obsData['T2S'] = obsData['T2S'] + \
                np.random.normal(loc=0,scale=obsData['T2S'][0]/self.SNR,size=np.shape(obsData['T2S']))

        return obsData

###############################################################################

    def createObsData_fast(self, MWF: float, addNoise=False):
        
        '''    
        Creates a MRI decay signal based on a Mean Water Fraction (MWF) value.

        Four signal types are possible to invert:
            T1 == obsData['T1'], T2 == obsData['T2'], T2S == obsData['T2S]
            
        The method is matrix based on integration over two gaussian curves (fast).
        
        !!! Method for matrix calculation is old (status: 11.2023) !!!
        
        Input parameters
        ----------
        MWF: mean water fraction value (type: float)
        
        addNoise: True or False (type: boolean)

        Returns: observed MRI relaxation data (type: numpy.ndarray)
        
        '''
        
        obsData = {'T1':[],'T2':[],'T2S':[]}
        
        if self.invT1==True:
            valMax         = self.Inversion.T1max
            valMin         = self.Inversion.T1min
            valMod         = self.Inversion.modSpace
            
            grid           = hlp.make_grid(valMin, valMax, valMod, mode='lin')
            A              = system_matrix_from_kernel(self.Inversion.T1_timepoints, grid, T1_decay_abs)
            
            values         = np.array([100,1000])
            weights        = np.array([MWF, 1-MWF])
            widths         = np.array([20, 100]) 
            
            m1             = hlp.gauss(grid, widths[0], values[0], weights[0])
            m2             = hlp.gauss(grid, widths[1], values[1], weights[1])
            
            m              = np.add(m1,m2)
            obsData['T1']  = np.matmul(A, m)*(valMax-valMin)/valMod

        if self.invT2==True:
            valMax         = self.Inversion.T2max
            valMin         = self.Inversion.T2min
            valMod         = self.Inversion.modSpace
            
            grid           = hlp.make_grid(valMin, valMax, valMod, mode='lin')
            EpyG_params    = EpyG_Paramset(self.Inversion.T2_T1,self.Inversion.T2_flipangle, 
                                        self.Inversion.T2_TE,self.Inversion.T2_ETL)
            
            A              = EpyG_mSE_system_matrix(EpyG_params, grid)

            values         = np.array([8, 80])
            weights        = np.array([MWF, 1-MWF]) # integrals of the Gaussians
            widths         = np.array([2, 8])       # standard deviation of the Gaussians
        
            m1             = hlp.gauss(grid, widths[0], values[0], weights[0])
            m2             = hlp.gauss(grid, widths[1], values[1], weights[1])
        
            m              = np.add(m1,m2)
            obsData['T2']  = np.matmul(A, m)*(valMax-valMin)/valMod

        if self.invT2S==True:
            valMax         = self.Inversion.T2Smax
            valMin         = self.Inversion.T2Smin
            valMod         = self.Inversion.modSpace
            
            grid           = hlp.make_grid(valMin, valMax, valMod, mode='lin')    
            A              = system_matrix_from_kernel(self.Inversion.T2S_timepoints, grid, T2_decay)
            
            values         = np.array([8, 80])
            weights        = np.array([MWF, 1-MWF]) # integrals of the Gaussians
            widths         = np.array([2, 8])       # standard deviation of the Gaussians
            
            m1             = hlp.gauss(grid, widths[0], values[0], weights[0])
            m2             = hlp.gauss(grid, widths[1], values[1], weights[1])
            
            m              = np.add(m1,m2)
            obsData['T2S'] = np.matmul(A, m)*(valMax-valMin)/valMod
            
        if addNoise:
            if self.invT1==True:
                np.random.seed(0)
                obsData['T1'] = obsData['T1'] + \
                np.random.normal(loc=0,scale=obsData['T1'][0]/self.SNR,size=np.shape(obsData['T1']))
            if self.invT2==True:
                np.random.seed(0)
                obsData['T2'] = obsData['T2'] + \
                np.random.normal(loc=0,scale=obsData['T2'][0]/self.SNR,size=np.shape(obsData['T2']))
            if self.invT2S==True:
                np.random.seed(0)
                obsData['T2S'] = obsData['T2S'] + \
                np.random.normal(loc=0,scale=obsData['T2S'][0]/self.SNR,size=np.shape(obsData['T2S']))
        
        return obsData
    
###############################################################################

    def createSynData_slow(self):

        sys.exit('This method is not implemented yet.')
        
        self.synDat[:,:,:] = np.asarray([self.T1_signal(self.MWF[ii, 0],
                                                        self.Inversion.T1min,
                                                        self.Inversion.T1max,
                                                        self.Inversion.T1_timepoints) for ii in range(self.noPart)])

###############################################################################

    def createSynData_fast(self, matrix_version):

        '''    
        Creates synthetic MRI decay signals from Mean Water Fraction (MWF) values.
        Signal curves are being embedded into numpy array for each PSO pixle.

        Three signal types can be calculated:
            T1  == self.synDat['T1']
            T2  == self.synDat['T2']
            T2S == self.synDat['T2S']
            
        The method is matrix based on integration over two gaussian curves (fast).
        
        Method for matrix calculation can be chosen:
            V1 - status: 11.2023
            V2 - status: 12.2023
        
        '''

        if matrix_version == 'V1':
            
            if self.invT1==True:
                valMax     = self.Inversion.T1max
                valMin     = self.Inversion.T1min
                valMod     = self.Inversion.modSpace
                
                grid       = hlp.make_grid(valMin, valMax, valMod, mode='lin')
                A          = system_matrix_from_kernel(self.Inversion.T1_timepoints, grid, T1_decay_abs)
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T1'][ii], self.m2['T1'][ii]])         # centers of the Gaussians
                    weights = np.array([self.MWF['T1'][ii], 1-self.MWF['T1'][ii]])     # integrals of the Gaussians
                    widths  = np.array([self.m1_sig['T1'][ii], self.m2_sig['T1'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(grid, widths[0], values[0], weights[0])
                    m2 = hlp.gauss(grid, widths[1], values[1], weights[1])
                    
                    m  = np.add(m1,m2)
                    
                    self.synDat['T1'][ii] = np.matmul(A, m)*(self.Inversion.T1max-self.Inversion.T1min)/self.Inversion.modSpace            
        
            if self.invT2==True:
                valMax      = self.Inversion.T2max
                valMin      = self.Inversion.T2min
                valMod      = self.Inversion.modSpace
                
                grid        = hlp.make_grid(valMin, valMax, valMod, mode='lin')
                EpyG_params = EpyG_Paramset(self.Inversion.T2_T1,self.Inversion.T2_flipangle, 
                                            self.Inversion.T2_TE,self.Inversion.T2_ETL)
                
                A           = EpyG_mSE_system_matrix(EpyG_params, grid)
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2'][ii],self.m2['T2'][ii]])         # centers of the Gaussians
                    weights = np.array([self.MWF['T2'][ii],1-self.MWF['T2'][ii]])     # integrals of the Gaussians
                    widths  = np.array([self.m1_sig['T2'][ii],self.m2_sig['T2'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(grid, widths[0], values[0], weights[0])
                    m2 = hlp.gauss(grid, widths[1], values[1], weights[1])
                    
                    m  = np.add(m1,m2)
                    
                    self.synDat['T2'][ii] = np.matmul(A, m)*(self.Inversion.T2max-self.Inversion.T2min)/self.Inversion.modSpace

            if self.invT2==True:
                valMax      = self.Inversion.T2Smax
                valMin      = self.Inversion.T2Smin
                valMod      = self.Inversion.modSpace
                
                grid        = hlp.make_grid(valMin, valMax, valMod, mode='lin')
                EpyG_params = EpyG_Paramset(self.Inversion.T2_T1,self.Inversion.T2_flipangle, 
                                            self.Inversion.T2_TE,self.Inversion.T2_ETL)
                
                A           = EpyG_mSE_system_matrix(EpyG_params, grid)
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2S'][ii],self.m2['T2S'][ii]])         # centers of the Gaussians
                    weights = np.array([self.MWF['T2S'][ii],1-self.MWF['T2S'][ii]])     # integrals of the Gaussians
                    widths  = np.array([self.m1_sig['T2S'][ii],self.m2_sig['T2S'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(grid, widths[0], values[0], weights[0])
                    m2 = hlp.gauss(grid, widths[1], values[1], weights[1])
                    
                    m  = np.add(m1,m2)
                    
                    self.synDat['T2S'][ii] = np.matmul(A, m)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace

                    
        if matrix_version == 'V2' and self.noPeaks == 'GAUSS':
                        
            if self.invT1==True:
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T1'][ii],self.m2['T1'][ii]])         # centers of the Gaussians
                    weights = np.array([self.int2['T1'][ii]*self.MWF['T1'][ii]/(1-self.MWF['T1'][ii]), 
                                        self.int2['T1'][ii]])
                    widths  = np.array([self.m1_sig['T1'][ii],self.m2_sig['T1'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T1'], widths[0], values[0], weights[0])
                    m2 = hlp.gauss(self.sysGrid['T1'], widths[1], values[1], weights[1])
                    
                    m  = np.add(m1,m2)
                    
                    self.synDat['T1'][ii] = np.matmul(self.sysMatrix['T1'], m)*(self.Inversion.T1max-self.Inversion.T1min)/self.Inversion.modSpace            
    
            if self.invT2==True:
                
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2'][ii],self.m2['T2'][ii]])         # centers of the Gaussians
                    weights = np.array([self.int2['T2'][ii]*self.MWF['T2'][ii]/(1-self.MWF['T2'][ii]), 
                                        self.int2['T2'][ii]])                         # integral of the Gaussians
                    widths  = np.array([self.m1_sig['T2'][ii],self.m2_sig['T2'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T2'], widths[0], values[0], weights[0])
                    m2 = hlp.gauss(self.sysGrid['T2'], widths[1], values[1], weights[1])
                    
                    m  = np.add(m1,m2)
                    
                    self.synDat['T2'][ii] = np.matmul(self.sysMatrix['T2'], m)*(self.Inversion.T2max-self.Inversion.T2min)/self.Inversion.modSpace
            
            if self.invT2S==True:
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2S'][ii],self.m2['T2S'][ii]])         # centers of the Gaussians
                    weights = np.array([self.int2['T2S'][ii]*self.MWF['T2S'][ii]/(1-self.MWF['T2S'][ii]), 
                                        self.int2['T2S'][ii]])                          # integral of the Gaussians
                    widths  = np.array([self.m1_sig['T2S'][ii],self.m2_sig['T2S'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T2S'], widths[0], values[0], weights[0]) # computation of the Gaussian curve m1
                    m2 = hlp.gauss(self.sysGrid['T2S'], widths[1], values[1], weights[1]) # computation of the Gaussian curve m2

                    m  = np.add(m1,m2)
                    
                    self.synDat['T2S'][ii,:] = np.matmul(self.sysMatrix['T2S'], m)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace


        if matrix_version == 'V2' and self.noPeaks == 'DIRAC':

            if self.invT1==True:
                
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T1'][ii],self.m2['T1'][ii],self.m3['T1'][ii]])             # centers of the Gaussians
                    weights = np.array([(self.int2['T1'][ii]+self.int3['T1'][ii])*self.MWF['T1'][ii]/(1-self.MWF['T1'][ii]), 
                                         self.int2['T1'][ii],self.int3['T1'][ii]])                          # integral of the Gaussians
                    widths  = np.array([self.m1_sig['T1'][ii],self.m2_sig['T1'][ii],self.m3_sig['T1'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T1'], widths[0], values[0], weights[0]) # computation of the Gaussian curve m1
                    m2 = hlp.gauss(self.sysGrid['T1'], widths[1], values[1], weights[1]) # computation of the Gaussian curve m2
                    m3 = hlp.gauss(self.sysGrid['T1'], widths[2], values[2], weights[2]) # computation of the Gaussian curve m3
                    
                    m  = np.add(m1,m2,m3)
                    
                    self.synDat['T1'][ii] = np.matmul(self.sysMatrix['T1'], m)*(self.Inversion.T1max-self.Inversion.T1min)/self.Inversion.modSpace
                    
            if self.invT2==True:
                
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2'][ii],self.m2['T2'][ii],self.m3['T2'][ii]])             # centers of the Gaussians
                    weights = np.array([(self.int2['T2'][ii]+self.int3['T2'][ii])*self.MWF['T2'][ii]/(1-self.MWF['T2'][ii]), 
                                         self.int2['T2'][ii],self.int3['T2'][ii]])                          # integral of the Gaussians
                    widths  = np.array([self.m1_sig['T2'][ii],self.m2_sig['T2'][ii],self.m3_sig['T2'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T2'], widths[0], values[0], weights[0]) # computation of the Gaussian curve m1
                    m2 = hlp.gauss(self.sysGrid['T2'], widths[1], values[1], weights[1]) # computation of the Gaussian curve m2
                    m3 = hlp.gauss(self.sysGrid['T2'], widths[2], values[2], weights[2]) # computation of the Gaussian curve m3
                    
                    m  = np.add(m1,m2,m3)
                    
                    self.synDat['T2'][ii] = np.matmul(self.sysMatrix['T2'], m)*(self.Inversion.T2max-self.Inversion.T2min)/self.Inversion.modSpace
            
            if self.invT2S==True:
    
                for ii in range(self.noPart):
                    values  = np.array([self.m1['T2S'][ii],self.m2['T2S'][ii],self.m3['T2S'][ii]])             # centers of the Gaussians
                    weights = np.array([(self.int2['T2S'][ii]+self.int3['T2S'][ii])*self.MWF['T2S'][ii]/(1-self.MWF['T2S'][ii]),
                                         self.int2['T2S'][ii],self.int3['T2S'][ii]])                           # integral of the Gaussians
                    widths  = np.array([self.m1_sig['T2S'][ii],self.m2_sig['T2S'][ii],self.m3_sig['T2S'][ii]]) # standard deviation of the Gaussians
                    
                    m1 = hlp.gauss(self.sysGrid['T2S'], widths[0], values[0], weights[0]) # computation of the Gaussian curve m1
                    m2 = hlp.gauss(self.sysGrid['T2S'], widths[1], values[1], weights[1]) # computation of the Gaussian curve m2
                    m3 = hlp.gauss(self.sysGrid['T2S'], widths[2], values[2], weights[2]) # computation of the Gaussian curve m3

                    m  = np.add(m1,m2,m3)
                    
                    self.synDat['T2S'][ii] = np.matmul(self.sysMatrix['T2S'], m)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace
            
            
###############################################################################
# Überarbeitung notwendig: V0 and L1 could be tested on both single/joint inv !

    def fitness(self):
        
        '''
        Calculation of misfit between observed and synthetic data.
        
        Parameters for use in PSO class initiation:
            optionInv [V0, V1] - method for joint inversion
                                 V0: one single model vector for each signal\n
                                 V1: one common model vector for both signals
            
            lpNorm [l1, L2] - Lp-Norm for generalization
                              L1: linear or L2: quadratic
                              
        Construction work: L1 for single inversion can be tested, beta.
                           V0 can be tested for lp-norm 2.
        '''
        
        if self.singleInv==True:
                
            sig = self.signType[0]
                
            for ii in range(0, self.noPart):
                
                if self.lpNorm == 'L1':
                    self.fit[sig][ii] = np.sum(np.abs(self.obsData[sig]-self.synDat[sig][ii]))/len(self.obsData[sig])

                if self.lpNorm == 'L2': # adjusted to Sego-Code: Frobenius Norm 
                    self.fit[sig][ii] = np.sqrt(np.sum((self.obsData[sig]-self.synDat[sig][ii])**2))/len(self.obsData[sig])


        if self.jointInv==True and self.optionInv=='V0':

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for ii in range(0, self.noPart):
                
                if self.lpNorm == 'L1':
                    sys.exit('ATTENTION: L1-Norm not yet implemeted for joint inversion.')
                    continue

                if self.lpNorm == 'L2':
                    term_sig_1      = np.sum((self.obsData[i]-self.synDat[i][ii,:])**2)
                    weight_alpha    = 2
                    term_sig_2      = np.sum((self.obsData[j]-self.synDat[j][ii,:])**2)
                    weight_beta     = 1
                    term_MWF        = (self.MWF[i][ii]-self.MWF[j][ii])**2
                    weight_gamma    = 1
                       
                    self.fit[k][ii] = weight_alpha  * np.sqrt(term_sig_1)/len(self.obsData[i])  + \
                                      weight_beta   * np.sqrt(term_sig_2)/len(self.obsData[j])  + \
                                      weight_gamma  * term_MWF
        
        if self.jointInv==True and self.optionInv=='V1':
            
            sys.exit('ATTENTION: Joint Inversion not implemented yet')
            
            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for ii in range(0, self.noPart):
                
                if self.lpNorm == 'L1':
                    sys.exit('ATTENTION: L1-Norm not yet implemeted for joint inversion.')
                    continue

                if self.lpNorm == 'L2':
                    term_sig_1      = np.sum((self.obsData[i]-self.synDat[i][ii,:])**2)/len(self.obsData[i])
                    weight_alpha    = 1
                    term_sig_2      = np.sum((self.obsData[j]-self.synDat[j][ii,:])**2)/len(self.obsData[j])
                    weight_beta     = 1
                       
                    self.fit[k][ii] = weight_alpha  * np.sqrt(term_sig_1)  + \
                                      weight_beta   * np.sqrt(term_sig_2)
                
###############################################################################

    def meanMisfit(self, data1: np.array, data2: np.array):
        
        if self.lpNorm == 'L1':
            return np.sum(np.abs(data1-data2))/len(data1)
    
        if self.lpNorm == 'L2':
            return np.sqrt(np.sum((data1-data2)**2)/len(data1))
            
###############################################################################            
    
    def bestLocal(self):

        '''
        Update of the best local position/parameters for each swarm particle.
        
        Parameters for use in PSO class initiation:
            optionInv [V0, V1] - method for joint inversion
                                 V0: one single model vector for each signal\n
                                 V1: one common model vector for both signals
        
        Construction work: V1 should be tested and updated, beta.
        '''

        fit    = copy.deepcopy(self.fit)
        bfit   = copy.deepcopy(self.bestFit)
        MWF    = copy.deepcopy(self.MWF)        
        mod    = copy.deepcopy(self.mod)
        synDat = copy.deepcopy(self.synDat)

        if self.singleInv==True:
            
            sig    = self.signType[0]
            
            self.bestFit[sig][fit[sig]<=bfit[sig]]    = fit[sig][fit[sig]    <= bfit[sig]]
            self.bestMWF[sig][fit[sig]<=bfit[sig]]    = MWF[sig][fit[sig]    <= bfit[sig]]            
            self.bestMod[sig][fit[sig]<=bfit[sig]]    = mod[sig][fit[sig]    <= bfit[sig]]
            self.bestSynDat[sig][fit[sig]<=bfit[sig]] = synDat[sig][fit[sig] <= bfit[sig]]

        if self.jointInv==True and self.optionInv=='V0':   

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for sig in self.signType:

                self.bestMWF[sig][fit[k]<=bfit[k]]    = MWF[sig][fit[k]    <= bfit[k]]
                self.bestMod[sig][fit[k]<=bfit[k]]    = mod[sig][fit[k]    <= bfit[k]]
                self.bestSynDat[sig][fit[k]<=bfit[k]] = synDat[sig][fit[k] <= bfit[k]]
                
            self.bestFit[k][fit[k]<=bfit[k]]          = fit[k][fit[k]    <= bfit[k]]
            self.bestMWF[k]                           = np.sum([self.bestMWF[i],self.bestMWF[j]], axis=0)/self.signal_cc

        if self.jointInv==True and self.optionInv=='V1':   

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for sig in self.signType:

                self.bestSynDat[sig][fit[k]<=bfit[k]] = synDat[sig][fit[k] <= bfit[k]]

            self.bestMWF[k][fit[k]<=bfit[k]]        = MWF[k][fit[k]    <= bfit[k]]
            self.bestMod[fit[k]<=bfit[k]]           = mod[fit[k]       <= bfit[k]]                
            self.bestFit[k][fit[k]<=bfit[k]]        = fit[k][fit[k]    <= bfit[k]]

###############################################################################

    def bestGlobal(self):

        '''
        Update of the best local position/parameters for each swarm particle.
        '''

        bfit      = copy.deepcopy(self.bestFit)

        if self.singleInv==True:
            
            sig       = self.signType[0]
            actMin    = {sig: np.min(bfit[sig])}
            actMinInd = {sig: np.argmin(bfit[sig])}
            
            if actMin[sig] < self.globFit[sig]:
                    self.globFit[sig]    = actMin[sig]
                    self.globMod[sig]    = np.copy(self.mod[sig][actMinInd[sig]])
                    self.globSynDat[sig] = np.copy(self.bestSynDat[sig][actMinInd[sig]])
                    self.globIndex[sig]  = actMinInd[sig]
        
        if self.jointInv==True and self.optionInv=='V0':
            
            globFit   = self.globFit[self.sigJoint]

            sig       = self.sigJoint
            actMin    = {sig: np.min(bfit[sig])}
            actMinInd = {sig: np.argmin(bfit[sig])}
            
            i,j,k     = self.signType[0], self.signType[1], self.sigJoint

            for sig in self.signType:
                
                if actMin[k] < globFit:
                    self.globFit[k]   = actMin[k]
                    self.globIndex[k] = actMinInd[k]
                    self.globMod[sig] = np.copy(self.mod[sig][actMinInd[k]])

        if self.jointInv==True and self.optionInv=='V1':
            
            sys.exit('V1 inversion not implemented yet.')

            sig       = self.sigJoint
            actMin    = {sig: np.min(bfit[sig])}
            actMinInd = {sig: np.argmin(bfit[sig])}
            
            if actMin[sig] < self.globFit[sig]:
                self.globFit[sig]   = actMin[sig]
                self.globIndex[sig] = actMinInd[sig]
                self.globMod[sig]   = np.copy(self.mod[3, actMinInd[sig]]) # !!! was ist index 3 ? 
                    
###############################################################################

    def updatePos(self):
        
        w  = self.PSO.w
        c1 = self.PSO.c1
        c2 = self.PSO.c2

        for sig in self.signType:

            for ii in range(self.noPart):

                r1 = np.random.rand(self.noParam)             # Intervall [0,1]
                r2 = np.random.rand(self.noParam)             # Intervall [0,1]

                self.vel[sig][ii] = w*self.vel[sig][ii] + \
                                    c1*r1*(self.bestMod[sig][ii]-self.mod[sig][ii]) + \
                                    c2*r2*(self.globMod[sig]-self.mod[sig][ii])
                                 
                self.mod[sig][ii] = self.mod[sig][ii] + self.vel[sig][ii]

###############################################################################
    
    def checkLim(self):

        for sig in self.signType:

            m1     = getattr(getattr(self, sig), self.noPeaks).m1
            m1_sig = getattr(getattr(self, sig), self.noPeaks).m1_sig
            m2     = getattr(getattr(self, sig), self.noPeaks).m2
            m2_sig = getattr(getattr(self, sig), self.noPeaks).m2_sig
            int2   = getattr(getattr(self, sig), self.noPeaks).int2
            MWF    = getattr(getattr(self, sig), self.noPeaks).MWF
            
            if self.noPeaks == 'GAUSS':
                
                for ii in range(self.noPart):
            
                    self.m1[sig][ii]     = np.clip(self.mod[sig][ii,0],m1[0],m1[1])
                    self.m1_sig[sig][ii] = np.clip(self.mod[sig][ii,1],m1_sig[0],m1_sig[1])
                    self.m2[sig][ii]     = np.clip(self.mod[sig][ii,2],m2[0],m2[1])
                    self.m2_sig[sig][ii] = np.clip(self.mod[sig][ii,3],m2_sig[0],m2_sig[1])
                    self.int2[sig][ii]   = np.clip(self.mod[sig][ii,4],int2[0],int2[1])
                    self.MWF[sig][ii]    = np.clip(self.mod[sig][ii,5],MWF[0],MWF[1])
                        
                    self.vel[sig][ii,0]  = 0 if self.m1[sig][ii]     != self.mod[sig][ii, 0] else self.vel[sig][ii,0]
                    self.vel[sig][ii,1]  = 0 if self.m1_sig[sig][ii] != self.mod[sig][ii, 1] else self.vel[sig][ii,1]
                    self.vel[sig][ii,2]  = 0 if self.m2[sig][ii]     != self.mod[sig][ii, 2] else self.vel[sig][ii,2]
                    self.vel[sig][ii,3]  = 0 if self.m2_sig[sig][ii] != self.mod[sig][ii, 3] else self.vel[sig][ii,3]
                    self.vel[sig][ii,4]  = 0 if self.int2[sig][ii]   != self.mod[sig][ii, 4] else self.vel[sig][ii,4]
                    self.vel[sig][ii,5]  = 0 if self.MWF[sig][ii]    != self.mod[sig][ii, 5] else self.vel[sig][ii,5]
        
                    self.mod[sig][ii]    = np.column_stack((self.m1[sig][ii],   self.m1_sig[sig][ii], 
                                                            self.m2[sig][ii],   self.m2_sig[sig][ii], 
                                                            self.int2[sig][ii], self.MWF[sig][ii]))
            
            if self.noPeaks == 'DIRAC':
                
                m3     = getattr(getattr(self, sig), self.noPeaks).m3
                m3_sig = getattr(getattr(self, sig), self.noPeaks).m3_sig
                int3   = getattr(getattr(self, sig), self.noPeaks).int3
                
                for ii in range(self.noPart):
                    
                    self.m1[sig][ii]     = np.clip(self.mod[sig][ii,0],m1[0],m1[1])
                    self.m1_sig[sig][ii] = np.clip(self.mod[sig][ii,1],m1_sig[0],m1_sig[1])
                    self.m2[sig][ii]     = np.clip(self.mod[sig][ii,2],m2[0],m2[1])
                    self.m2_sig[sig][ii] = np.clip(self.mod[sig][ii,3],m2_sig[0],m2_sig[1])
                    self.m3[sig][ii]     = np.clip(self.mod[sig][ii,4],m3[0],m3[1])
                    self.m3_sig[sig][ii] = np.clip(self.mod[sig][ii,5],m3_sig[0],m3_sig[1])
                    self.int2[sig][ii]   = np.clip(self.mod[sig][ii,6],int2[0],int2[1])
                    self.int3[sig][ii]   = np.clip(self.mod[sig][ii,7],int3[0],int3[1])
                    self.MWF[sig][ii]    = np.clip(self.mod[sig][ii,8],MWF[0],MWF[1])
                        
                    self.vel[sig][ii,0]  = 0 if self.m1[sig][ii]     != self.mod[sig][ii, 0] else self.vel[sig][ii,0]
                    self.vel[sig][ii,1]  = 0 if self.m1_sig[sig][ii] != self.mod[sig][ii, 1] else self.vel[sig][ii,1]
                    self.vel[sig][ii,2]  = 0 if self.m2[sig][ii]     != self.mod[sig][ii, 2] else self.vel[sig][ii,2]
                    self.vel[sig][ii,3]  = 0 if self.m2_sig[sig][ii] != self.mod[sig][ii, 3] else self.vel[sig][ii,3]
                    self.vel[sig][ii,4]  = 0 if self.m3[sig][ii]     != self.mod[sig][ii, 4] else self.vel[sig][ii,4]
                    self.vel[sig][ii,5]  = 0 if self.m3_sig[sig][ii] != self.mod[sig][ii, 5] else self.vel[sig][ii,5]
                    self.vel[sig][ii,6]  = 0 if self.int2[sig][ii]   != self.mod[sig][ii, 6] else self.vel[sig][ii,6]
                    self.vel[sig][ii,7]  = 0 if self.int3[sig][ii]   != self.mod[sig][ii, 7] else self.vel[sig][ii,7]
                    self.vel[sig][ii,8]  = 0 if self.MWF[sig][ii]    != self.mod[sig][ii, 8] else self.vel[sig][ii,8]
        
                    self.mod[sig][ii]    = np.column_stack((self.m1[sig][ii],   self.m1_sig[sig][ii], 
                                                            self.m2[sig][ii],   self.m2_sig[sig][ii],
                                                            self.m3[sig][ii],   self.m3_sig[sig][ii],
                                                            self.int2[sig][ii], self.int3[sig][ii], self.MWF[sig][ii]))                                
  
###############################################################################

    def init_MWF_Analysis(self, rootMWF, noSlice):
        
        # rootMWF.prep_data(axis        = 'Z', 
        #                   slice_num   = noSlice, 
        #                   signal_type = 'T1',
        #                   filter      = 1.0, 
        #                   thresh      = 4.5, 
        #                   verbose     = False)

        rootMWF.prep_t1_model(tr      = self.Inversion.T1_TR,
                              alpha   = self.Inversion.T1_alpha,
                              td      = self.Inversion.T1_TD,
                              ie      = self.Inversion.T1_IE,
                              T1min   = self.Inversion.T1min,
                              T1max   = self.Inversion.T1max,
                              nT1     = self.Inversion.modSpace,
                              verbose = False)
        
        rootMWF.prep_data(axis        = 'Z',
                          slice_num   = noSlice,
                          signal_type = 'T2',
                          filter      = 1.0,
                          thresh      = 4.5,
                          verbose     = False)

        rootMWF.prep_t2_model(te      = self.Inversion.T2_TE,
                              tr      = self.Inversion.T2_TR,
                              etl     = self.Inversion.datSpaceT2,
                              alpha   = self.Inversion.T2_alpha,
                              beta    = self.Inversion.T2_beta,
                              T2min   = self.Inversion.T2min,
                              T2max   = self.Inversion.T2max,
                              nT2     = self.Inversion.modSpace,
                              T1      = self.Inversion.T2_T1,
                              verbose = False)
        
        rootMWF.prep_data(axis        = 'Z', 
                          slice_num   = noSlice, 
                          signal_type = 'T2S',
                          filter      = 1.0, 
                          thresh      = 4.5, 
                          verbose     = False)

        rootMWF.prep_t2s_model(T2Smin  = self.Inversion.T2Smin,
                                T2Smax  = self.Inversion.T2Smax,
                                nT2S    = self.Inversion.modSpace,
                                verbose = False)

###############################################################################

    def init_sysMatrix(self, rootMWF, **kwargs):
        
        yy, xx, noSlice = kwargs.get('position', (0,0,0))
        
        n_fs = np.argmin(np.abs(rootMWF.b1_grid - rootMWF.data.slice['B1'][yy,xx]))

        self.sysMatrix['T1']  = rootMWF.sm['T1'][:,:,0]
        self.sysMatrix['T2']  = rootMWF.sm['T2'][:,:,n_fs]
        self.sysMatrix['T2S'] = rootMWF.sm['T2S'][:,:,0]
        
###############################################################################
    
    def init_Grid(self):
        
        self.sysGrid['T1']    = hlp.make_grid(self.Inversion.T1min, 
                                              self.Inversion.T1max,
                                              self.Inversion.modSpace, mode='lin')

        self.sysGrid['T2']    = hlp.make_grid(self.Inversion.T2min, 
                                              self.Inversion.T2max,
                                              self.Inversion.modSpace, mode='lin')

        self.sysGrid['T2S']   = hlp.make_grid(self.Inversion.T2Smin, 
                                              self.Inversion.T2Smax,
                                              self.Inversion.modSpace, mode='lin')

###############################################################################

    def write_CSV(self):
        print('Writing CSV-file is not implemented yet.')
        return
    
# ##############################################################################
# ## writing raw PSO data into a csv-table

# if singleInv==True and writeData==True:
    
#     for i in range(array_length):
        
#         if (i==0 and not invT1) or (i==1 and not invT2) or (i==2 and not invT2star): 
#             continue
        
#         typeSig = ['T1', 'T2', 'T2_'][i]

#         df_MWF = pd.DataFrame({f'MWF_{item:.3f}': MWF for item, MWF in zip(MWF_intervall, MWF_mult[:,i])})
#         df_fit = pd.DataFrame({f'Fit_{item:.3f}': Fit for item, Fit in zip(MWF_intervall, Fit_mult[:,i])})
#         df_PSO = pd.concat([df_MWF, df_fit], axis=1)
    
#         os.makedirs(path_save, exist_ok=True)
#         df_PSO.to_csv(f'{path_save}/data_{typeSig}{strSAVE}.csv', index=False)

# if jointInv==True and writeData==True:

#     df_MWF = pd.DataFrame({f'MWF_{item:.3f}': MWF for item, MWF in zip(MWF_intervall, MWF_mult[:,3])})
#     df_fit = pd.DataFrame({f'Fit_{item:.3f}': Fit for item, Fit in zip(MWF_intervall, Fit_mult[:,3])})
#     df_PSO = pd.concat([df_MWF, df_fit], axis=1)

#     os.makedirs(path_save, exist_ok=True)
#     df_PSO.to_csv(f'{path_save}/data_{typeSig}{strSAVE}.csv', index=False)

###############################################################################
    
    def write_JSON(self, savepath: str):
        
        if self.singleInv==True:
            
            if self.noPeaks == 'GAUSS':
                print('JSON-writer not implemented for GAUSS model parameters yet.')
                print(input('Continue?'))
                pass
                
            savepath    = f'{savepath}lvl1/{self.noPeaks}/{self.signType[0]}_param.json'
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            
            datspace    = self.Inversion.datSpaceT2 if self.signType[0]=='T2' else self.Inversion.datSpaceT2S
            
            jsonData    = {"PSO specifications":{
                                "_Signal type": self.signType[0],
                                "_Iterations":  self.noIter,
                                "_Particles":   self.noPart,
                                
                                
                                "_PSO cycles":  self.noPSOIter,
                                "_w":           self.PSO.w,
                                "_c1":          self.PSO.c1,
                                "_c2":          self.PSO.c2,
                                "_Lp Norm":     self.lpNorm},
                            "ObsData":{
                                "_Solver":               self.dataType,
                                "_No steps integration": self.Inversion.modSpace,
                                "_Noise":                np.nan,
                                "_Noiselevel":           np.nan,
                                "_timepoints":           datspace},
                            f"SynData{self.signType[0]}":{
                                "_Solver": 'matrix-based',
                                "_Model":  self.noPeaks,
                                "_m1":     getattr(getattr(self, self.signType[0]), self.noPeaks).m1,
                                "_m1_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m1_sig, 
                                "_m2":     getattr(getattr(self, self.signType[0]), self.noPeaks).m2,
                                "_m2_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m2_sig, 
                                "_m3":     getattr(getattr(self, self.signType[0]), self.noPeaks).m3,
                                "_m3_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m3_sig,
                                "_integ2": getattr(getattr(self, self.signType[0]), self.noPeaks).int2,
                                "_integ3": getattr(getattr(self, self.signType[0]), self.noPeaks).int3,
                                "_MWF":    getattr(getattr(self, self.signType[0]), self.noPeaks).MWF
                            }}
        
            f = json.dumps(jsonData).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                            
            with open(savepath, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
        if self.jointInv==True:

            if self.noPeaks == 'GAUSS':
                print('JSON-writer not implemented for GAUSS model parameters yet.')
                print(input('Continue?'))
                pass
                
            savepath    = f'{savepath}lvl1/{self.noPeaks}/{self.signType[0]}{self.signType[1]}_param.json'
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            
            jsonData    = {"PSO specifications":{
                                "_Signal type": self.signType,
                                "_Iterations":  self.noIter,
                                "_Particles":   self.noPart,
                                "_PSO cycles":  self.noPSOIter,
                                "_w":           self.PSO.w,
                                "_c1":          self.PSO.c1,
                                "_c2":          self.PSO.c2,
                                "_Lp Norm":     self.lpNorm},
                            "ObsData":{
                                "_Solver":               self.dataType,
                                "_No steps integration": self.Inversion.modSpace,
                                "_Noise":                np.nan,
                                "_Noiselevel":           np.nan,
                                "_timepointsT2":         self.Inversion.datSpaceT2,
                                "_timepointsT2S":        self.Inversion.datSpaceT2S},
                            "SynDataT2":{
                                "_Solver": 'matrix-based',
                                "_Model":  self.noPeaks,
                                "_m1":     getattr(getattr(self, self.signType[0]), self.noPeaks).m1,
                                "_m1_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m1_sig, 
                                "_m2":     getattr(getattr(self, self.signType[0]), self.noPeaks).m2,
                                "_m2_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m2_sig, 
                                "_m3":     getattr(getattr(self, self.signType[0]), self.noPeaks).m3,
                                "_m3_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m3_sig,
                                "_integ2": getattr(getattr(self, self.signType[0]), self.noPeaks).int2,
                                "_integ3": getattr(getattr(self, self.signType[0]), self.noPeaks).int3,
                                "_MWF":    getattr(getattr(self, self.signType[0]), self.noPeaks).MWF},
                            "SynDataT2S":{
                                "_Solver": 'matrix-based',
                                "_Model":  self.noPeaks,
                                "_m1":     getattr(getattr(self, self.signType[1]), self.noPeaks).m1,
                                "_m1_sig": getattr(getattr(self, self.signType[1]), self.noPeaks).m1_sig, 
                                "_m2":     getattr(getattr(self, self.signType[1]), self.noPeaks).m2,
                                "_m2_sig": getattr(getattr(self, self.signType[1]), self.noPeaks).m2_sig, 
                                "_m3":     getattr(getattr(self, self.signType[1]), self.noPeaks).m3,
                                "_m3_sig": getattr(getattr(self, self.signType[1]), self.noPeaks).m3_sig,
                                "_integ2": getattr(getattr(self, self.signType[1]), self.noPeaks).int2,
                                "_integ3": getattr(getattr(self, self.signType[1]), self.noPeaks).int3,
                                "_MWF":    getattr(getattr(self, self.signType[1]), self.noPeaks).MWF
                            }}

            f = json.dumps(jsonData).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                        
            with open(savepath, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
###############################################################################
    
    def result2array(self,results:dict,result_map:np.array,kk:int,
                     cutThresh=(0.45,0.005,False),cutMask=(np.array,False),calcBestResult=False):
    
        results_ = [i for i in results if i != None]
            
        for sig in self.signType:
            
            for line in results_:
                yy,xx = line['pix'][0],line['pix'][1]
                
                result_map[sig][-1,yy,xx,kk] = line['fit']
        
                for ii,item in enumerate(line[f'mod{sig}']):
                    result_map[sig][ii,yy,xx,kk] = item        
        
                if kk == self.noPSOIter-1 and calcBestResult==True:
                    result_map[sig] = self.__bestfit2array__(result_map[sig],
                                                             position=(yy,xx))
            
            if cutThresh[2]==True:
                result_map[sig] = self.__cutarray2thresh__(result_map[sig],
                                                           cutThresh[0:2],
                                                           cut2thresh=True)            
            if cutMask[1]==True:
                result_map[sig] = self.__cutarray2mask__(result_map[sig],
                                                         cutMask[0],
                                                         cut2mask=True)
                            
            # result_map[-1,yy,xx,kk] = line['fit']
            
            # for ii,item in enumerate(line['mod']):
            #     result_map[ii,yy,xx,kk] = item
            
            # if kk == self.noPSOIter-1 and calcBestResult==True:
            #     result_map = self.__bestfit2array__(result_map, position=(yy,xx))
            
            # if cutThresh[2]==True:
            #     result_map = self.__cutarray2thresh__(result_map,cutThresh[0:2],cut2thresh=True)
            
            # if cutMask[1]==True:
            #     result_map = self.__cutarray2mask__(result_map, cutMask[0])
        
        return result_map
        
###############################################################################

    def __bestfit2array__(self, result_map: np.array, position: tuple):
        
        yy,xx = position[0], position[1]
        
        try:
            bestFitID              = np.nanargmin(result_map[-1,yy,xx,:-1])
            result_map[:,yy,xx,-1] = result_map[:,yy,xx,bestFitID]
        except:
            result_map[:,yy,xx,-1] = np.nan     
        
        return result_map

###############################################################################

    def __cutarray2thresh__(self, result_map: np.array, threshold: tuple, cut2thresh=False):

        if cut2thresh==False:
            return result_map
        
        threshMWF, threshFit = threshold[0], threshold[1]
        
        result_map[0][np.logical_and(result_map[-2]>threshMWF,
                                     result_map[-1]>threshFit)]=np.nan
        
        for i in range(len(result_map)):
            result_map[i][np.isnan(result_map[0])]=np.nan
        
        return result_map

###############################################################################

    def __cutarray2mask__(self, result_map: np.array, mask: np.array, cut2mask=False):

        if cut2mask==False:
            return result_map
        
        for i in range(result_map.shape[-1]):
            for j in range(result_map.shape[0]):
                result_map[j,:,:,i][np.isnan(mask[0,:,:,0])]=np.nan
        
        return result_map
    
###############################################################################
    
    # def filter_nan_neighbors(result_map, threshMWF, threshFit):
        
    #     # Finden Sie die Nachbarschaftsstruktur für 4 Nachbarn (oben, unten, links, rechts)
    #     structure = generate_binary_structure(2, 1)
    
    #     # Identifizieren Sie zusammenhängende Bereiche von np.nan für threshMWF und threshFit
    #     labeled_nan, _ = label(result_map[0][np.logical_and(result_map[-2]>threshMWF,
    #                                  result_map[-1]>threshFit)], structure=structure)
    
    #     # Iterieren Sie über alle Pixel und setzen Sie np.nan, wenn die Bedingungen erfüllt sind
    #     for i in range(result_map.shape[1]):
    #         for j in range(result_map.shape[2]):
    #             if np.logical_and(result_map[-2, i, j] > threshMWF, result_map[-1, i, j] > threshFit):
    #                 region_indices = np.where(labeled_nan == labeled_nan[i, j])
    #                 nan_neighbors = np.sum(np.isnan(result_map[region_indices]))
    #                 if nan_neighbors <= 1:
    #                     result_map[:, i, j, :] = np.nan
    
    #     return result_map
            
###############################################################################    
        
    def initPS(self, matrix_version):
        
        # solving the forward modelling for each particle
        self.createSynData_fast(matrix_version)
        
        # evaluation of the swarm fitness
        self.fitness()

        # save best values for every particle
        self.bestLocal()

        # save global best values
        self.bestGlobal()
        
        # collect best global fit and indicies for a number of n iterations
        if self.plotResult==True and self.singleInv==True:
            
            sig                    = self.signType[0]
            self.globInd_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.globFit_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.globMWF_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.globInd_list[sig] = [self.globIndex[sig]+1]
            self.globFit_list[sig] = [self.globFit[sig]]
            self.globMWF_list[sig] = [self.globMod[sig][-1]]
        
###############################################################################  
    
    def execPSO(self, obsData: np.array, matrix_version: str,
                indPix: tuple, PSOiter: int,
                savepath: str, string=None, plotResult=False, plotIter=False):
        
        # some object and parameter definitions before
        # --> especially grid and A needs some time for being computed
        if plotResult==False: self.plotResult=False
        if plotResult==True:  self.plotResult=True
        
        if plotIter==False:   self.plotIter=False
        if plotIter==True:    self.plotIter=True

        self.obsData = obsData
        self.initPS(matrix_version)
        
        for jj in range(self.noIter)[:]:
            
            # print(f'\nexecute iteration number {jj+1}')
            # print(f'global best fit, pix. {self.globIndex}: {self.globFit}')

            # update particle position in the global search space
            # each Particle  -->  locVector = w*v1 + c1*r1*(m[i+1]-m[i])              
            self.updatePos()
            # print(self.vel)
            
            # # check the limits of new model values
            # # --> using better np.clip() funtion instead of generic expressions            
            self.checkLim()
            # # print(self.vel)
    
            # update synthetic data set
            self.createSynData_fast(matrix_version)
            # print(self.synDat[2])
                        
            # update particle fitness
            self.fitness()
            # print(self.fit)
            
            # update local best for each particle
            self.bestLocal()
            # print(self.bestFit[2])
      
            # update global best for the particle swarm
            self.bestGlobal()
            # print(self.globIndex[2])

            # collect best global fit and indicies for a number of n iterations
            if plotResult==True and self.singleInv==True:
                sig = self.signType[0]
                self.globInd_list[sig].append(self.globIndex[sig]+1)
                self.globFit_list[sig].append(self.globFit[sig])
                self.globMWF_list[sig].append(self.globMod[sig][-1])
                    
            #     for i in range(3):
            #         if (i==0 and not self.invT1) or (i==1 and not self.invT2) or (i==2 and not self.invT2star): 
            #             continue
                    
            #         if jj==self.PSO.noIter-1:
            #             self.plotPSO_result(data=data, indPlot=jj, indPix=indPix, indSig=i,
            #                                 PSOiter=PSOiter, savepath=savepath, string=string)

###############################################################################

    def log(self, startTime, string='', dim='sek', boolean=False):
        
        '''
        Parameters
        ----------
        startTime : time.time() object

        string : string

        dim : output diension
              ms  - milli seconds
              mus - micro seconds
              MS  - min:sec (default)
              HMS - hr:min:sec.

        Returns : None
        '''
        
        if boolean == True:
            print(boolean)
            
        T_now       = time.time()
        T_elapsed   = T_now - startTime
        
        if dim=='HMS':
            TT = time.strftime('%H:%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} hrs')
            
        if dim=='MS':
            TT = time.strftime('%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} min')
        
        if dim=='ms':
            t_ms        = round(T_elapsed*1000, 2)
            print(f'{string}: {t_ms} ms')
        
        if dim=='mus':
            t_mus        = round(T_elapsed*1000 * 1000, 2)
            print(f'{string}: {t_mus} \u03BCs')
            
###############################################################################    

class PSO_Plots():

    def __init__(self):
        pass   
        
    def plotPixel(self,
                  MWFclass: object,
                  PSOclass: object,
                  position: tuple,
                  savepath: str,
                  string:   str):

        sig           = PSOclass.signType[0]
        yy,xx,noSlice = position[0], position[1], position[2]
        savepath      = f'{savepath}/{string}/'

        fig, ax       = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), tight_layout=True)
        
        ax[0,0].imshow(MWFclass.data.msk, cmap='gray')
        ax[0,0].scatter(xx, yy, color='red')
        ax[0,0].set_xticks([]); ax[0,0].set_yticks([])
        ax[0,0].set_title(f'MeasGrid for Pixel [{yy},{xx}] in Slice {noSlice}')

        ax[0,1].plot(MWFclass.tsig[sig], PSOclass.obsData[sig], 'k', linewidth=2)
        ax[0,1].plot(MWFclass.tsig[sig], PSOclass.globSynDat[sig], markersize=2, linestyle='-', 
                     marker='o', markeredgewidth=0.5, color='red', linewidth=2, markerfacecolor='lightpink')
        ax[0,1].set_title(f'{sig}: observed (black) vs. synthetic (red) signal')
        ax[0,1].set_ylim(np.min(PSOclass.obsData[sig]-0.1), np.max(PSOclass.obsData[sig])+0.1)

        ax[1,0].plot(np.arange(0, len(PSOclass.globInd_list[sig]), 1), PSOclass.globInd_list[sig], 
                     markersize=2, linestyle='None', marker='o', color='b')
        ax[1,0].set_ylim(0.5,  PSOclass.noPart+0.5)
        ax[1,0].set_xlim(-0.5, PSOclass.noIter+0.5)
        ax[1,0].set_xticks(np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5)), 
                           np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5), dtype=int))
        ax[1,0].set_yticks(np.arange(0,PSOclass.noPart+1,PSOclass.noPart/10))
        ax[1,0].set_title('global best particle')   
        
        ylim_min = np.min(PSOclass.globFit_list[sig])-np.min(PSOclass.globFit_list[sig])/3
        ylim_max = np.max(PSOclass.globFit_list[sig])+np.min(PSOclass.globFit_list[sig])/3

        ax[1,1].plot(np.arange(0, len(PSOclass.globFit_list[sig]), 1), 
                     PSOclass.globFit_list[sig], markersize=2, linestyle='-', marker='o', color='b')
        ax[1,1].set_ylim(ylim_min, ylim_max)
        ax[1,1].set_xlim(-5, PSOclass.noIter+5)
        ax[1,1].set_xticks(np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5)),
                           np.arange(0, PSOclass.noIter+1, int(PSOclass.noIter/5), dtype=int))       
        # ax[1,1].set_yticks(np.arange(ylim_min, ylim_max, steps))

        ax[1,1].set_title('global best fit')
             
        value = PSOclass.globFit[sig]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'

        text  = (f'glob best [px: fit]:\n{PSOclass.globIndex[sig]+1}: {value}\n\n'
                  f'MWFcalc:{np.round(PSOclass.globMod[sig][-1], 4)}')
        
        x_lim = (PSOclass.noIter+1)/10*7
        ax[1,1].text(x_lim, ylim_max/10*9.5, text, va='top')

        os.makedirs(savepath, exist_ok=True)
        plt.savefig(f'{savepath}pix_y-{yy}_x-{xx}.jpg', dpi=300, format='jpg')                
    
        plt.show(); plt.close()

###############################################################################

    def plotSliceSI(self, 
                    PSOclass:  object,
                    PSOresult: np.array,
                    index:     int,
                    saveFig:   tuple,
                    string =   ''):
        
      
        if PSOclass.signType[0] == 'T2' and PSOclass.noPeaks == 'GAUSS':
            limits = [PSOclass.T2.GAUSS.m1, (0,1), PSOclass.T2.GAUSS.m2, PSOclass.T2.GAUSS.m2_sig, 
                      PSOclass.T2.GAUSS.int2, (0,0.35), (0,0.005)]
        
        if PSOclass.signType[0] == 'T2' and PSOclass.noPeaks == 'DIRAC':
            limits = [PSOclass.T2.DIRAC.m1, (0,1), PSOclass.T2.DIRAC.m2, (0,1), PSOclass.T2.DIRAC.m3, 
                      (0,1), PSOclass.T2.DIRAC.int2, PSOclass.T2.DIRAC.int3, (0,0.25), (0,0.005)]           
        
        if PSOclass.signType[0] == 'T2S' and PSOclass.noPeaks == 'GAUSS':
            limits = [PSOclass.T2S.GAUSS.m1, (0,1), PSOclass.T2S.GAUSS.m2, PSOclass.T2S.GAUSS.m2_sig, 
                      PSOclass.T2S.GAUSS.int2, (0,0.35), (0,0.005)]

        if PSOclass.signType[0] == 'T2S' and PSOclass.noPeaks == 'DIRAC':
            limits = [PSOclass.T2S.DIRAC.m1, (0,1), PSOclass.T2S.DIRAC.m2, (0,1), PSOclass.T2S.DIRAC.m3, 
                      (0,1), PSOclass.T2S.DIRAC.int2, PSOclass.T2S.DIRAC.int3, (0,0.25), (0,0.005)]

        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        
        # GAUSS results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: int2 | -2: MWF   | -1: misfit
        # DIRAC results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: m3   | 5: m3_sig | 
        #                 6: int2 | 7: int3   | -2: MWF | -1: misfit
        im0 = ax[0,0].imshow(PSOresult[-2,:,:,index],cmap=cmap,vmin=limits[-2][0],vmax=limits[-2][1])
        ax[0,0].set_title('MWF')
        fig.colorbar(im0, ax=ax[0,0])
        
        im1 = ax[0,1].imshow(PSOresult[0,:,:,index],cmap=cmap,vmin=limits[0][0],vmax=limits[0][1])
        ax[0,1].set_title('m1')
        fig.colorbar(im1, ax=ax[0,1])
        
        im2 = ax[0,2].imshow(PSOresult[2,:,:,index],cmap=cmap,vmin=limits[2][0],vmax=limits[2][1])
        ax[0,2].set_title('m2')
        fig.colorbar(im2, ax=ax[0,2])
        
        im3 = ax[1,0].imshow(PSOresult[-1,:,:,index],cmap=cmap,vmin=limits[-1][0],vmax=limits[-1][1])
        ax[1,0].set_title('misfit')
        fig.colorbar(im3, ax=ax[1,0])
        
        if PSOclass.noPeaks == 'GAUSS':
            im4 = ax[1,1].imshow(PSOresult[1,:,:,index],cmap=cmap,vmin=limits[1][0],vmax=limits[1][1])
            ax[1,1].set_title('m1_sig')
            fig.colorbar(im4, ax=ax[1,1])
        
        if PSOclass.noPeaks == 'DIRAC':
            im4 = ax[1,1].imshow(PSOresult[4,:,:,index],cmap=cmap,vmin=limits[4][0],vmax=limits[4][1])
            ax[1,1].set_title('m3')
            fig.colorbar(im4, ax=ax[1,1])
        
        if PSOclass.noPeaks == 'GAUSS':
            im5 = ax[1,2].imshow(PSOresult[3,:,:,index],cmap=cmap,vmin=limits[3][0],vmax=limits[3][1])
            ax[1,2].set_title('m2_sig')
            fig.colorbar(im5, ax=ax[1,2])

        if PSOclass.noPeaks == 'DIRAC':
            im4 = ax[1,2].imshow(PSOresult[5,:,:,index],cmap=cmap,vmin=limits[5][0],vmax=limits[5][1])
            ax[1,2].set_title('m3_sig')
            fig.colorbar(im4, ax=ax[1,2])
        
        for ii in range(ax.shape[0]*ax.shape[1]):
            ax.flat[ii].axis('off')
            ax.flat[ii].set_facecolor('b')
            fig.set_facecolor('w')
        
        if index == PSOclass.noPSOIter or index == -1: 
            print('bfit'); index = 'bfit'

        dist = '3 peaks (DIRAC)' if PSOclass.noPeaks == 'DIRAC' else '2 peaks (GAUSS)'
        
        fig.suptitle(f'Calculated {PSOclass.signType[0]} map for {dist}. PSO iteration: {str(index).zfill(2)}')
        
        if saveFig[1]==True:
            savepath = f'{saveFig[0]}{string}/'
            os.makedirs(savepath, exist_ok=True)
            fig.savefig(f'{savepath}ID_{str(index).zfill(2)}.png', dpi=300, format='png')

        plt.show(); plt.close()

###############################################################################

    def plotSliceJI(self, 
                    PSOclass:  object,
                    PSOresult: dict,
                    index:     int,
                    saveFig:   tuple,
                    string =   '',
                    **kwargs):
        
        param     = kwargs.get('limit', (object,False))[0]
        paramBool = kwargs.get('limit', (object,False))[1]
        lim       = {sig:[] for sig in PSOclass.signType}
        
        if paramBool == False:            
            if PSOclass.noPeaks == 'GAUSS':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1,(0,1),
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int2,
                                (0,0.35), (0,0.005)]
            
            if PSOclass.noPeaks == 'DIRAC':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m1_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m2_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m3,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).m3_sig,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int2,
                                getattr(getattr(PSOclass, sig), PSOclass.noPeaks).int3,
                                (0,0.25), (0,0.005)]
        
        if paramBool == True:
            if PSOclass.noPeaks == 'GAUSS':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],(0,1),
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['integ2'],(0,0.35), (0,0.005)]  
        
            if PSOclass.noPeaks == 'DIRAC':                
                for sig in PSOclass.signType:                    
                    lim[sig] = [param[f'SynData{sig}']['m1'],param[f'SynData{sig}']['m1_sig'],
                                param[f'SynData{sig}']['m2'],param[f'SynData{sig}']['m2_sig'],
                                param[f'SynData{sig}']['m3'],param[f'SynData{sig}']['m3_sig'],
                                param[f'SynData{sig}']['integ2'],param[f'SynData{sig}']['integ3'],
                                (0,0.25), (0,0.005)]        

        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        for sig in PSOclass.signType:
            
            PSOres  = PSOresult[sig]  
            _lim    = lim[sig]
            ind     = index
            
            if PSOclass.noPeaks=='GAUSS':
                fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            
            if PSOclass.noPeaks=='DIRAC':
                fig, ax = plt.subplots(2, 4, figsize=(14, 6))
        
        # GAUSS results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: int2 | -2: MWF   | -1: misfit
        # DIRAC results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: m3   | 5: m3_sig | 
        #                 6: int2 | 7: int3   | -2: MWF | -1: misfit
            im0 = ax[0,0].imshow(PSOres[-2,:,:,ind],cmap=cmap,vmin=_lim[-2][0],vmax=_lim[-2][1])
            ax[0,0].set_title('MWF')
            fig.colorbar(im0, ax=ax[0,0])
        
            im1 = ax[0,1].imshow(PSOres[0,:,:,ind],cmap=cmap,vmin=_lim[0][0],vmax=_lim[0][1])
            ax[0,1].set_title('m1')
            fig.colorbar(im1, ax=ax[0,1])
        
            im2 = ax[0,2].imshow(PSOres[2,:,:,ind],cmap=cmap,vmin=_lim[2][0],vmax=_lim[2][1])
            ax[0,2].set_title('m2')
            fig.colorbar(im2, ax=ax[0,2])
            
            im3 = ax[1,0].imshow(PSOres[-1,:,:,ind],cmap=cmap,vmin=_lim[-1][0],vmax=_lim[-1][1])
            ax[1,0].set_title('misfit')
            fig.colorbar(im3, ax=ax[1,0])

            im4 = ax[1,1].imshow(PSOres[1,:,:,ind],cmap=cmap,vmin=_lim[1][0],vmax=_lim[1][1])
            ax[1,1].set_title('m1_sig')
            fig.colorbar(im4, ax=ax[1,1])
            
            im5 = ax[1,2].imshow(PSOres[3,:,:,ind],cmap=cmap,vmin=_lim[3][0],vmax=_lim[3][1])
            ax[1,2].set_title('m2_sig')
            fig.colorbar(im5, ax=ax[1,2])
        
            if PSOclass.noPeaks == 'DIRAC':
                im6 = ax[0,3].imshow(PSOres[4,:,:,ind],cmap=cmap,vmin=_lim[4][0],vmax=_lim[4][1])
                ax[0,3].set_title('m3')
                fig.colorbar(im6, ax=ax[0,3])

                im7 = ax[1,3].imshow(PSOres[5,:,:,ind],cmap=cmap,vmin=_lim[5][0],vmax=_lim[5][1])
                ax[1,3].set_title('m3_sig')
                fig.colorbar(im7, ax=ax[1,3])
        
            for ii in range(ax.shape[0]*ax.shape[1]):
                ax.flat[ii].axis('off')
                ax.flat[ii].set_facecolor('b')
                fig.set_facecolor('w')
        
            if ind == PSOclass.noPSOIter or ind == -1: 
                print('bfit'); ind = 'bfit'

            dist = '3 peaks (DIRAC)' if PSOclass.noPeaks == 'DIRAC' else '2 peaks (GAUSS)'
        
            fig.suptitle(f'Calculated {sig} map for {dist}. PSO iteration: {str(ind).zfill(2)}')
        
            if saveFig[1]==True:
                savepath = f'{saveFig[0]}{string}/'
                os.makedirs(savepath, exist_ok=True)
                fig.savefig(f'{savepath}{sig}_ID_{str(ind).zfill(2)}.png', dpi=300, format='png')

            plt.show(); plt.close()

###############################################################################

    def plotHist(self,
                 PSOclass:     object,
                 PSOresult:    np.array,
                 PSOresultCUT: np.array,
                 position:     tuple,
                 noBins:       int,
                 saveFig:      tuple,
                 string =      ''):
        
        yy,xx    = position[0], position[1]
        res2     = np.copy(PSOresultCUT)
        
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 10), tight_layout=True)
        cmap = matplotlib.colormaps.get_cmap('viridis')
        cmap.set_bad(color='black')
        
        im0=ax[0,0].imshow(PSOresult[-2,:,:,-1],cmap=cmap,vmin=0,vmax=0.35)
        ax[0,0].scatter(yy,xx, color='red',marker='o')
        ax[0,0].set_title(f'MWF (best fit: {np.round(PSOresult[-2,yy,xx,-1],3)})')
        fig.colorbar(im0, ax=ax[0,0])

        hist, bins = np.histogram(PSOresult[-2, yy, xx, :-1], bins=noBins, density=False)
        ax[0,1].hist(PSOresult[-2, yy, xx, :-1], bins=noBins, edgecolor='black')
        ax[0,1].vlines(np.max(res2[-2]),ymin=0,ymax=np.max(hist), linestyle='dashed',
                       linewidth=1, color='r', label=f'MWF cut: {np.round(np.max(res2[-2]),3)}')
        ax[0,1].legend(loc='upper right')
        
        ax[0,2].hist(res2[-2,:-1], bins=noBins, edgecolor='black',label='noVal')
        # ax[0,2].legend(loc='upper right')
                
        value = PSOresult[-1,yy,xx,-1]; n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'
                
        im1=ax[1,0].imshow(PSOresult[-1,:,:,-1],cmap=cmap,vmin=0,vmax=0.005)
        ax[1,0].scatter(yy,xx, color='red',marker='o')
        ax[1,0].set_title(f'misfit (best fit: {value})')
        fig.colorbar(im1, ax=ax[1,0])

        hist, bins = np.histogram(PSOresult[-1, yy, xx, :-1], bins=noBins, density=False)
        
        value = np.max(res2[-1]); n = 0
        while abs(value) < 1:
            value *= 10
            n -= 1
        rounded_number = round(value, 3)
        value = f'{rounded_number} * 10e{n}'
        
        ax[1,1].hist(PSOresult[-1, yy, xx, :-1], bins=noBins, edgecolor='black',label=f'no.Values: {res2.shape[-1]}|{PSOresult.shape[-1]-1}')
        ax[1,1].vlines(bins[1],ymin=0,ymax=np.max(hist), linestyle='dashed',
                        linewidth=1, color='r', label=f'misfit cut: {value}')
        ax[1,1].legend(loc='upper right')
        
        ax[1,2].hist(res2[-1,:-1], bins=noBins, edgecolor='black',label=f'no.Values: {res2.shape[-1]}|{PSOresult.shape[-1]-1}')
        # ax[1,2].legend(loc='upper right')
        
        vmin=getattr(getattr(PSOclass, PSOclass.signType[0]), PSOclass.noPeaks).m1[0]
        vmax=getattr(getattr(PSOclass, PSOclass.signType[0]), PSOclass.noPeaks).m1[1]
        
        im2=ax[2,0].imshow(PSOresult[0,:,:,-1],cmap=cmap,vmin=vmin,vmax=vmax)
        ax[2,0].scatter(yy,xx, color='red',marker='o')
        ax[2,0].set_title(f'm1 (best fit: {np.round(PSOresult[0,yy,xx,-1],1)})')
        fig.colorbar(im2, ax=ax[2,0])
        
        hist, bins = np.histogram(PSOresult[0, yy, xx, :-1], bins=noBins, density=False)
        ax[2,1].hist(PSOresult[0, yy, xx, :-1], bins=noBins, edgecolor='black')
        ax[2,1].vlines(np.max(res2[0]),ymin=0,ymax=np.max(hist), linestyle='dashed', 
                       linewidth=1, color='r', label=f'm1 cut: {np.round(np.max(res2[0]),3)}') 
        ax[2,1].legend(loc='upper right')
        ax[2,1].set_xlim(vmin, vmax)
        
        ax[2,2].hist(res2[0,:-1], bins=noBins, edgecolor='black',label='ID bfit[0]')
        # ax[2,2].legend(loc='upper right')
        
        ax.flat[0].axis('off')
        ax.flat[0].set_facecolor('b')
        ax.flat[3].axis('off')
        ax.flat[3].set_facecolor('b')
        ax.flat[6].axis('off')
        ax.flat[6].set_facecolor('b')
        fig.set_facecolor('w')

        fig.suptitle(f'Pix.[{yy},{xx}]: Calculated parameters for signal {PSOclass.signType[0]} | {PSOclass.noPeaks}',fontsize=14)
        
        if saveFig[1]==True:
            savepath = f'{saveFig[0]}{string}\\'
            os.makedirs(savepath, exist_ok=True) 
            plt.savefig(f'{savepath}pix_y-{yy}_x-{xx}.png', dpi=300, format='png')  
            
        plt.show; plt.close()


###############################################################################

    def plotMWFvsFIT(self,
                     PSOclass:     object,
                     PSOresult:    np.array,
                     position:     tuple,
                     saveFig:      tuple,
                     string =      '',
                     **kwargs):

        # get keyword arguments from the function call
        cutPercentile  = kwargs.get('cutPercentile', False)
        valPercentile  = kwargs.get('valPercentile', None)
        
        cutOutliers    = kwargs.get('cutOutliers', False)
        valOutliers    = kwargs.get('valOutliers', None)

        # pixel position        
        yy,xx        = position[0], position[1]        
        MWFarray     = PSOresult[-2,yy,xx,:-1]
        FITarray     = PSOresult[-1,yy,xx,:-1]

        # plot raw
        fig, ax = plt.subplots(tight_layout=True)        
        fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
        
        ax.plot(FITarray, MWFarray, markersize=2, linestyle='none', 
                marker='o', markeredgewidth=0.5, color='r', markerfacecolor='lightpink')
        
        ax.set_title(f'{PSOclass.signType[0]} @ Pix.[{yy},{xx}] | {PSOclass.noIter}Iter,'
                     f'{PSOclass.noPart}Part,{PSOclass.noPSOIter}PSO | {PSOclass.noPeaks}')
        
        ax.set_ylabel('global best MWF []'); plt.xlabel('global best Fit []')
        
        if saveFig[1]==True:
            savepath = f'{saveFig[0]}{string}\\'
            os.makedirs(savepath, exist_ok=True) 
            plt.savefig(f'{savepath}MWFvsFIT_pix_y-{yy}_x-{xx}.png', dpi=300, format='png')
            
        plt.show; plt.close()

        # cut outliers and percentile
        if cutPercentile == True:
                
            # calculate percentiles and plot the curve of global bestMWF vs. global bestFit
            outliers = np.where((FITarray > np.percentile(FITarray, valPercentile[1])))
            
            MWFarray = np.delete(MWFarray, outliers)
            FITarray = np.delete(FITarray, outliers)        
        
        
        if cutOutliers == True:            
            median       = np.median(FITarray)
            stdv         = np.std(FITarray)
    
            MWFarray     = MWFarray[FITarray<median+stdv*valOutliers[1]] # upper              
            FITarray     = FITarray[FITarray<median+stdv*valOutliers[1]] # upper

        # plot cut            
        fig, ax = plt.subplots(tight_layout=True)        
        fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
        
        ax.plot(FITarray, MWFarray, markersize=2, linestyle='none', 
                marker='o', markeredgewidth=0.5, color='r', markerfacecolor='lightpink')
        
        ax.set_title(f'{PSOclass.signType[0]} @ Pix.[{yy},{xx}] | {PSOclass.noIter}Iter,'
                     f'{PSOclass.noPart}Part,{PSOclass.noPSOIter}PSO | {PSOclass.noPeaks}')
        
        ax.set_ylabel('global best MWF []'); plt.xlabel('global best Fit []')
        
        if saveFig[1]==True:
            savepath = f'{saveFig[0]}{string}\\'
            os.makedirs(savepath, exist_ok=True) 
            plt.savefig(f'{savepath}MWFvsFIT_pix_y-{yy}_x-{xx}_cut.png', dpi=300, format='png')
            
        plt.show; plt.close()

            
###############################################################################

    def plotPSO_iter(self, 
                     data: tuple, 
                     savepath: str, 
                     meta: list,
                     *args,
                     **kwargs):

        '''
        Function for plotting (bestFit vs. bestMWF) as single grafic for each PSO Iteration
        
        Parameters
        ----------

        data:          format tuple (x, y)
        
        savepath:      path to the storage directory (will be created if not exists)
        
        meta:          [type of signal, number of PSO reps, MWF value]
        
        args:          Additional positional arguments, e.g., not yet
        
        kwargs:        Keyword arguments, e.g., valPercentile=(lower, upper), dataMANIP=string, posHLine=string
                       Boolean arguments, e.g., cutPercentile, cutOutliers, clustDBSCAN
        '''
        
        # get keyword arguments from the function call
        cutPercentile  = kwargs.get('cutPercentile', False)
        valPercentile  = kwargs.get('valPercentile', None)
        
        cutOutliers    = kwargs.get('cutOutliers', False)
        valOutliers    = kwargs.get('valOutliers', None)
        
        clustDBSCAN    = kwargs.get('clustDBSCAN', False)
        valDBSCAN      = kwargs.get('valDBSCAN', None)
           
        posHLine       = kwargs.get('posHLine', None)
        dataMANIP      = kwargs.get('dataMANIP', '')
        meanMisfit     = kwargs.get('meanMisfit', None)
        axesLimit      = kwargs.get('axesLimit', '')
        addNoise       = kwargs.get('addNoise', False)

        # convert list data to numpy array
        MWF_array = np.array(data[0])
        Fit_array = np.array(data[1])

        # # drop values, that are being set as edges of MWF intervall (0.02, 0.75)         
        # Fit_calc  = Fit_array[(MWF_array!=self.T2.MWF[0])&(MWF_array!=self.T2.MWF[1])]
        # MWF_calc  = MWF_array[(MWF_array!=self.T2.MWF[0])&(MWF_array!=self.T2.MWF[1])]
        
        MWF_calc = MWF_array
        Fit_calc = Fit_array

        # --> eliminate percentiles
        # --> threshold: median +- double standard deviation        
        if cutPercentile == True:
            
            lower    = valPercentile[0]
            upper    = valPercentile[1]
                
            # calculate percentiles and plot the curve of global bestMWF vs. global bestFit
            outliers = np.where((MWF_calc > np.percentile(MWF_calc, upper)) |
                                (MWF_calc < np.percentile(MWF_calc, lower)) |
                                (Fit_calc > np.percentile(Fit_calc, upper)) |
                                (Fit_calc < np.percentile(Fit_calc, lower)))
            
            MWF_calc = np.delete(MWF_calc, outliers)
            Fit_calc = np.delete(Fit_calc, outliers)
         
        else:
            MWF_calc = MWF_calc
            Fit_calc = Fit_calc
        
        # --> eliminate obvious outliers in both directions
        # --> thresold: median +- double standard deviation 
        if cutOutliers == True:
            
            median   = np.median(MWF_calc)
            stdv     = np.std(MWF_calc)
            
            Fit_calc = Fit_calc[MWF_calc<median+stdv*valOutliers[0]] # upper
            MWF_calc = MWF_calc[MWF_calc<median+stdv*valOutliers[0]] # upper            
            Fit_calc = Fit_calc[MWF_calc>median-stdv*valOutliers[0]] # lower
            MWF_calc = MWF_calc[MWF_calc>median-stdv*valOutliers[0]] # lower
            
            median   = np.median(Fit_calc)
            stdv     = np.std(Fit_calc)
    
            MWF_calc = MWF_calc[Fit_calc<median+stdv*valOutliers[1]] # upper              
            Fit_calc = Fit_calc[Fit_calc<median+stdv*valOutliers[1]] # upper          
            # MWF_calc = MWF_calc[Fit_calc>median-stdv*valOutliers[1]] # lower 
            # Fit_calc = Fit_calc[Fit_calc>median-stdv*valOutliers[1]] # lower
          
        
        # if clustDBSCAN == True and len(MWF_calc)!=0:
            
        #     data                  = np.array(list(zip(Fit_calc, MWF_calc)))

        #     best_eps              = None
        #     best_min_samples      = None
        #     best_silhouette_score = -1

        #     # for eps in np.arange(0.00001, 0.001, 0.00001):
        #     #     for min_samples in np.arange(100,200,2):
        #     #         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        #     #         labels = dbscan.fit_predict(data)
        #     #         if len(set(labels)) > 1:  # Ensure more than one cluster is found
        #     #             silhouette = silhouette_score(data, labels)
        #     #             if silhouette > best_silhouette_score:
        #     #                 best_silhouette_score = silhouette
        #     #                 best_eps = eps
        #     #                 best_min_samples = min_samples

        #     # # print(f'\nMWF: {meta[2]}')
        #     # print("Best eps:", best_eps)
        #     # print("Best min_samples:", best_min_samples)
        #     # print("Best Silhouette Score:", best_silhouette_score)
            
        #     if best_eps == None:
        #         best_eps=valDBSCAN[0]
        #         best_min_samples=valDBSCAN[1]
            
        #     dat_DBSC = list(zip(Fit_calc, MWF_calc))
        #     dbscan   = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        #     labels   = dbscan.fit_predict(dat_DBSC)
            
        #     unique_labels, label_counts = np.unique(labels, return_counts=True)

        #     # labels_dummy  = np.arange(-1,5)
        #     # labels_count  = np.zeros(6)
        #     # labels_index  = np.where(np.isin(labels_dummy, unique_labels))[0]
        #     # for i, ind in enumerate(labels_index):
        #     #     labels_count[ind] = label_counts[i]
        #     # string = [f'({labels_dummy[i]}: {int(labels_count[i])})' for i in range(len(labels_dummy))]
            
        #     labels_dummy  = np.arange(-1,6)
        #     labels_count  = np.bincount(labels+1, minlength=6)[labels_dummy]
        #     labels_count  = np.concatenate((labels_count[1:], np.array([0])))
            
        #     fig, ax = plt.subplots(tight_layout=True)            
        #     fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
            
        #     for i, label in enumerate(labels_dummy[:-3]):
        #         MWF_ = MWF_calc[labels == label]
        #         Fit_ = Fit_calc[labels == label]

        #         # color = plt.cm.viridis(label / len(unique_labels))  # CM of your choice
        #         color_dots = ['purple', 'red', 'green', 'blue', 'black', 'brown', 'yellow']
        #         color_mark = ['lavender', 'lightpink', 'lightgreen', 'lightblue', 'lightgray', 'wheat', 'lightyellow']
                
        #         ax.plot(Fit_, MWF_, linestyle='none', marker='o', markersize=2, 
        #                 markeredgewidth=0.5, color=color_dots[i], markerfacecolor=color_mark[i], 
        #                 label=f'count {labels_dummy[i]}: {int(labels_count[i])} ')
                            
        #     ax.set_title(f'{meta[0]} | {self.noIter}Iter,{self.noPart}Part,{meta[1]}PSO | '
        #                  f'MWFstart: {str(meta[2])[:5]}')
            
        #     ax.set_ylabel('global best MWF []'); plt.xlabel('global best Fit [%]')

        #     xmin, xmax = ax.get_xlim()        
        #     ax.hlines(meta[2], xmin=xmin, xmax=xmax, linestyle='dashed', linewidth=0.5, color='black')
            
        #     # y-lim abhängig von der höchsten Differenz
        #     diffmax  = np.max(abs(MWF_calc-float(meta[2])))
        #     ax.set_ylim(meta[2]-diffmax*(1+1/100), meta[2]+diffmax*(1+1/100))

        #     ax.legend(loc='lower right')
            
        #     # calculate a representative value for the "nose"
        #     # ideas: median, best global fit
        #     xmax_short = np.min(Fit_calc)
        #     yval_short = MWF_calc[np.argmin(Fit_calc)]
            
        #     ax.hlines(yval_short, xmin=xmin, xmax=xmax_short, linestyle='dashed', linewidth=0.5, color='black')
            
        #     os.makedirs(savepath, exist_ok=True)
        #     fig.savefig(f'{savepath}{self.noIter}I{self.noPart}P{str(meta[2])}M{dataMANIP}_col.jpg', 
        #                 dpi=300, format='jpg')
            
        #     plt.close()
            
        #     # # eliminate noise, which has DBSCAN-key: -1
        #     # Fit_calc = Fit_calc[labels!=-1] if len(Fit_calc[labels==-1]) <= 10 else Fit_calc
        #     # MWF_calc = MWF_calc[labels!=-1] if len(MWF_calc[labels==-1]) <= 10 else MWF_calc

        #     stop_exe = True
            
        #     if stop_exe == True:
        #         return np.median(MWF_calc)
        
        # figure plot
        # --> also considerable the use of is plt.scatter(x, y, color='r')
        fig, ax = plt.subplots(tight_layout=True)
        
        fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
        
        ax.plot(Fit_calc, MWF_calc, markersize=2, linestyle='none', 
                marker='o', markeredgewidth=0.5, color='r', markerfacecolor='lightpink')
        
        ax.set_title(f'{meta[0]} | {self.noIter}Iter,{self.noPart}Part,{meta[1]}PSO | '
                      f'MWFstart: {str(meta[2])[:5]}')
        
        ax.set_ylabel('global best MWF []'); plt.xlabel('global best Fit []')
      
        
        diffmax            = np.max(abs(MWF_calc-float(meta[2])))
        ymin_raw, ymax_raw = meta[2]-diffmax-diffmax/20, meta[2]+diffmax+diffmax/20
            
        # calculate a representative value for the "nose"
        # ideas: median, best global fit value
        
        try:
            xmin_short = ax.get_xlim()[0]
            xmax_short = np.min(Fit_calc)
            yval_short = MWF_calc[np.argmin(Fit_calc)]
        except:
            xmin_short = ax.get_xlim()[0]
            xmax_short = ax.get_xlim()[1]
            yval_short = meta[2] + 0.2*meta[2]

        # deviation of MWF value at best fit value from the starting MWF        
        dev_in_percent     = np.abs(meta[2]-yval_short)*100/meta[2]; n = 0
        dev_in_percent_val = np.copy(dev_in_percent)
        dev_in_absolut     = np.abs(meta[2]-yval_short)
        dev_in_absolut_val = np.copy(dev_in_absolut)
        dev_in_real        = yval_short - meta[2]
        
        while abs(dev_in_absolut) < 1:
            dev_in_absolut *= 10
            n -= 1
        rounded_number     = np.round(dev_in_absolut, 3)
        dev_in_absolut_str = f'{rounded_number}*10e{n}'        
        fit_best_abs       = np.copy(xmax_short); n = 0
        
        while abs(fit_best_abs) < 1:
            fit_best_abs *= 10
            n -= 1
        rounded_number     = np.round(fit_best_abs, 3)
        fit_best_abs_str   = f'{rounded_number}*10e{n}'

        # Grafic: cut, hline
        if meanMisfit==None and posHLine=='centered':
            ax.set_ylim(ymin_raw, ymax_raw)
            ax.hlines(yval_short, xmin=xmin_short, xmax=xmax_short, linestyle='dashed', linewidth=0.5, color='black')
            ax.hlines(meta[2], xmin=xmin_short, xmax=ax.get_xlim()[1], linestyle='dashed', linewidth=0.5, color='black')
            
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            xmax_txt = xmax - (xmax-xmin)/15
            ymax_txt = meta[2] + (ymax-ymin)/10*0.1
            ax.text(xmax_txt, ymax_txt,
                    f'dev {dev_in_absolut_str}', fontsize=10, color='black', ha='right')
        
        if meanMisfit==None and posHLine==None and axesLimit=='min_max':
            
            # noise - false: zz = 20
            # noise - true : zz = 2000
            
            zz        = 20
            
            x_diffmax = np.max(abs(np.min(Fit_calc)-np.max(Fit_calc)))
            y_diffmax = np.max(abs(np.min(MWF_calc)-np.max(MWF_calc)))            
            ax.set_ylim(np.min(MWF_calc)-y_diffmax/zz, np.max(MWF_calc)+y_diffmax/zz)
            ax.set_xlim(np.min(Fit_calc)-x_diffmax/zz, np.max(Fit_calc)+x_diffmax/zz)
            # ax.set_ylim(np.min(MWF_calc), np.max(MWF_calc))
            # ax.set_xlim(np.min(Fit_calc), np.max(Fit_calc))

        # Grafic: vline
        if meanMisfit!=None and posHLine=='non_specific' and addNoise==True: 
            
            if ax.get_xlim()[0] >= meanMisfit:
                diff = abs(meanMisfit-ax.get_xlim()[1])
                xmin = meanMisfit-diff/50
                xmax = ax.get_xlim()[1]+diff/50  
                
            elif ax.get_xlim()[1] <= meanMisfit:
                diff = abs(meanMisfit-ax.get_xlim()[0])
                xmin = ax.get_xlim()[0]-diff/50
                xmax = meanMisfit+diff/50 
                
            else: 
                xmin, xmax = ax.get_xlim()
                
            ax.hlines(meta[2], xmin=xmin, xmax=xmax, linestyle='dashed', linewidth=0.5, color='black')
            ax.hlines(yval_short, xmin=xmin, xmax=xmax_short, linestyle='dashed', linewidth=0.5, color='black')


            if ax.get_ylim()[0] >= meta[2]: 
                diff = abs(meta[2]-ax.get_ylim()[0])
                ymin = meta[2]-diff/50
                ymax = ax.get_ylim()[1]+diff/50
                
            elif ax.get_ylim()[1] <= meta[2]: 
                diff = abs(meta[2]-ax.get_ylim()[0])
                ymin = ax.get_ylim()[0]-diff/50
                ymax = meta[2]+diff/50
            
            else:
                ymin, ymax = ax.get_ylim()
                
            ax.vlines(meanMisfit, ymin=ymin, ymax=ymax, linestyle='solid', linewidth=1, color='grey')            

            ax.text(xmax-(xmax-xmin)/15, ymin+2*(ymax-ymin)/7, f'bestGlobalF: {np.round(xmax_short, 4)}', ha='right')
            ax.text(xmax-(xmax-xmin)/15, ymin+1.5*(ymax-ymin)/7, f'meanMisfit: {np.round(meanMisfit, 4)}', va='bottom', ha='right')
            ax.text(xmax-(xmax-xmin)/15, ymin+(ymax-ymin)/7, f'$\Delta$MWF: {dev_in_absolut_str}', va='bottom', ha='right')
        
        if meanMisfit!=None and posHLine==None and addNoise==True:
            
            if ax.get_xlim()[0] >= meanMisfit:
                diff = abs(meanMisfit-ax.get_xlim()[1])
                xmin = meanMisfit-diff/50
                xmax = ax.get_xlim()[1]+diff/50
                
            elif ax.get_xlim()[1] <= meanMisfit:
                diff = abs(meanMisfit-ax.get_xlim()[0])
                xmin = ax.get_xlim()[0]-diff/50
                xmax = meanMisfit+diff/50
                
            else: 
                xmin, xmax = ax.get_xlim()

            if ax.get_ylim()[0] >= meta[2]: 
                diff = abs(meta[2]-ax.get_ylim()[0])
                ymin = meta[2]-diff/50
                ymax = ax.get_ylim()[1]+diff/50
                
            elif ax.get_ylim()[1] <= meta[2]: 
                diff = abs(meta[2]-ax.get_ylim()[0])
                ymin = ax.get_ylim()[0]-diff/50
                ymax = meta[2]+diff/50
            
            else:
                ymin, ymax = ax.get_ylim()
                
            ax.vlines(meanMisfit, ymin=ymin, ymax=ymax, linestyle='solid', linewidth=1, color='grey')

            ax.text(xmax-(xmax-xmin)/15, ymin+2*(ymax-ymin)/7, f'bestGlobalF: {np.round(xmax_short, 3)}', ha='right')
            ax.text(xmax-(xmax-xmin)/15, ymin+1.5*(ymax-ymin)/7, f'meanMisfit: {np.round(meanMisfit, 3)}', va='bottom', ha='right')
                    
        os.makedirs(savepath, exist_ok=True)
        fig.savefig(f'{savepath}{self.noIter}I{self.noPart}P{str(meta[2])}M{dataMANIP}.jpg', 
                    dpi=300, format='jpg')
        
        plt.close()
        
        dev_in_percent_val = 10 if dev_in_percent_val > 10 else dev_in_percent_val
        
        return [dev_in_real, dev_in_absolut_val, dev_in_absolut_str, 
                fit_best_abs_str, np.round(yval_short,5)]

###############################################################################

    def plotPSO_mult(self,
                     data: tuple, 
                     savepath: str, 
                     meta: list,
                     *args,
                     **kwargs):
        
        '''
        Function for plotting (bestFit vs. bestMWF) as grafic consisting all PSO Iteration results
        
        Parameters
        ----------

        data:          format tuple (x, y)
        
        savepath:      path to the storage directory (will be created if not exists)
        
        meta:          [type of Signal, number of PSO reps, addNoise boolean]
        
        args:          Additional positional arguments, e.g., not yet
        
        kwargs:        Keyword arguments, e.g., cutPercentile=(lower, upper), dataMANIP=string
                       Boolean arguments, e.g., cutOutliers
        
        '''
        
        # get keyword arguments from the function call
        cutPercentile  = kwargs.get('cutPercentile', False)
        valPercentile  = kwargs.get('valPercentile', None)
        
        cutOutliers    = kwargs.get('cutOutliers', False)
        valOutliers    = kwargs.get('valOutliers', None)
        
        clustDBSCAN    = kwargs.get('clustDBSCAN', False)
        valDBSCAN      = kwargs.get('valDBSCAN', None)
           
        dataMANIP      = kwargs.get('dataMANIP', '')
        meanMisfit     = kwargs.get('meanMisfit', None)
        addNoise       = kwargs.get('addNoise', False)
        addNoise       = True if addNoise == 'True' else False
        
        # color            = plt.cm.RdBu(np.linspace(0, 1, len(data[0])))
        color            = np.linspace([1,0,0], [0,0,1], len(data[0]))
        
        fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)            
        fig.gca().xaxis.set_major_formatter('{:.3f}'.format)
            
        for i in range(len(data[0])):                
            
            MWF_array = np.copy(np.array(data[0][i]))
            Fit_array = np.copy(np.array(data[1][i]))
            
            MWF_calc = MWF_array
            Fit_calc = Fit_array
            
            # # drop values, that are being set as edges of MWF intervall (0.02, 0.75)         
            # Fit_calc  = Fit_array[(MWF_array!=self.T2.MWF[0])&(MWF_array!=self.T2.MWF[1])]
            # MWF_calc  = MWF_array[(MWF_array!=self.T2.MWF[0])&(MWF_array!=self.T2.MWF[1])]
                  
            # --> eliminate obvious outliers
            # --> thresold: median +- double standard deviation 
            if cutOutliers == True:
                
                median   = np.median(MWF_calc)
                stdv     = np.std(MWF_calc)
                
                Fit_calc = Fit_calc[MWF_calc<median+stdv*valOutliers[0]] # upper
                MWF_calc = MWF_calc[MWF_calc<median+stdv*valOutliers[0]] # upper            
                Fit_calc = Fit_calc[MWF_calc>median-stdv*valOutliers[0]] # lower
                MWF_calc = MWF_calc[MWF_calc>median-stdv*valOutliers[0]] # lower
                
                median   = np.median(Fit_calc)
                stdv     = np.std(Fit_calc)
    
                MWF_calc = MWF_calc[Fit_calc<median+stdv*valOutliers[1]] # upper              
                Fit_calc = Fit_calc[Fit_calc<median+stdv*valOutliers[1]] # upper          
                # MWF_calc = MWF_calc[Fit_calc>median-stdv*valOutliers[1]] # lower 
                # Fit_calc = Fit_calc[Fit_calc>median-stdv*valOutliers[1]] # lower
                
            if cutPercentile == True:
                
                lower    = valPercentile[0]
                upper    = valPercentile[1]
                    
                # calculate percentiles and plot the curve of global bestMWF vs. global bestFit
                outliers = np.where((MWF_calc > np.percentile(MWF_calc, upper)) |
                                    (MWF_calc < np.percentile(MWF_calc, lower)) |
                                    (Fit_calc > np.percentile(Fit_calc, upper)) |
                                    (Fit_calc < np.percentile(Fit_calc, lower)))
                
                MWF_calc = np.delete(MWF_calc, outliers)
                Fit_calc = np.delete(Fit_calc, outliers)
            
            if clustDBSCAN == True and len(MWF_calc)!=0 and meta[0]!='T2':
                
                dat_DBSC = list(zip(Fit_calc, MWF_calc))
                dbscan   = DBSCAN(eps=valDBSCAN[0], min_samples=valDBSCAN[1])
                labels   = dbscan.fit_predict(dat_DBSC)
                
                unique_labels, label_counts = np.unique(labels, return_counts=True)
            
                # eliminate noise, which has DBSCAN-key: -1
                Fit_calc = Fit_calc[labels!=-1]
                MWF_calc = MWF_calc[labels!=-1]

            # figure plot
            # --> also considerable the use of is plt.scatter(x, y, color='r')
            plt.gca().xaxis.set_major_formatter('{:.3f}'.format)

            ax.plot(Fit_calc, MWF_calc, markersize=1.25, linestyle='none', 
                     marker='o', markeredgewidth=0.2, color=color[i], 
                     markerfacecolor='lightpink', alpha=1)
            
            ax.set_title(f'{meta[0]} | {self.noIter}Iter,{self.noPart}Part,{meta[1]}PSO | MWFstart: overview')
            ax.set_ylabel('global best MWF []'); ax.set_xlabel('global best Fit [%]')   
        

        if addNoise==False:
            xmin, xmax = ax.get_xlim()
            [plt.hlines(i, xmin=xmin, xmax=xmax, linestyle='dashed', linewidth=0.5) for i in np.arange(0.025, 0.65, 0.05)]
        
        elif meanMisfit and addNoise==True:
            
            if ax.get_xlim()[0] >= np.min(np.array(meanMisfit)):
                diff = abs(np.min(np.array(meanMisfit))-ax.get_xlim()[1])
                xmin = np.min(np.array(meanMisfit))-diff/15
                xmax = ax.get_xlim()[1]+diff/15   
                
            elif ax.get_xlim()[1] <= np.max(np.array(meanMisfit)):
                diff = abs(np.max(np.array(meanMisfit))-ax.get_xlim()[0])
                xmin = ax.get_xlim()[0]-diff/15
                xmax = np.max(np.array(meanMisfit))+diff/15   
                
            else: 
                xmin, xmax = ax.get_xlim()
                
            [plt.hlines(i, xmin=xmin, xmax=xmax, linestyle='dashed', linewidth=0.5) for i in np.arange(0.025, 0.65, 0.05)]                
            plt.plot(meanMisfit, np.linspace(0.015, 0.65, len(meanMisfit)), linestyle='solid', linewidth=0.5, color='black')
            
        os.makedirs(savepath, exist_ok=True)        
        fig.savefig(f'{savepath}{meta[0]}_TOTAL{dataMANIP}.jpg', dpi=900, format='jpg')            
        plt.close()        


###############################################################################

    def log(self, startTime, string='', dim='sek', boolean=False):
        
        '''
        Parameters
        ----------
        startTime : time.time() object

        string : string

        dim : output diension
              ms  - milli seconds
              mus - micro seconds
              MS  - min:sec (default)
              HMS - hr:min:sec.

        Returns : None
        '''
        
        if boolean == True:
            print(boolean)
            
        T_now       = time.time()
        T_elapsed   = T_now - startTime
        
        if dim=='HMS':
            TT = time.strftime('%H:%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} hrs')
            
        if dim=='MS':
            TT = time.strftime('%M:%S', time.gmtime(T_elapsed))
            print(f'{string}: {TT} min')
        
        if dim=='ms':
            t_ms        = round(T_elapsed*1000, 2)
            print(f'{string}: {t_ms} ms')
        
        if dim=='mus':
            t_mus        = round(T_elapsed*1000 * 1000, 2)
            print(f'{string}: {t_mus} \u03BCs')
