# -*- coding: utf-8 -*-
"""
Class and methods for applying particle swarm optimizing (PSO) on invivo MRI data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)
"""

import time, sys, os, copy, json
import numpy             as     np
from   scipy.ndimage     import gaussian_filter

import helpTools         as     hlp
from   PSOparameters     import Parameters   as PM



class ParticleSwarmOptimizer(PM):
        
    '''
    Input parameters
        noPart: number of particles\n
        noIter: number of iterations for data fitting\n
        noPSOIter: number of complete PSO cycles\n
        signal_input: array containing signal\n
        **kwargs

    Returns: 
        observed MRI relaxation data (type: numpy.ndarray)
    '''
        
    def __init__(self,
                 noPart,
                 noIter,
                 noPSOIter,
                 signal_input,
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
        self.invOpt        = kwargs.get('invOpt',   'V0')
        self.noPeaks       = kwargs.get('noPeaks',  'GAUSS')
        self.modParam      = kwargs.get('modParam', ('class', None, None))
        
        if self.randSeedBool == True:
            np.random.seed(self.randSeedValue)

        # General parameters
        self.invT1     = signal_input[1][0]
        self.invT2     = signal_input[1][1]
        self.invT2S    = signal_input[1][2]
        self.dataType  = signal_input[1][3]
        self.addNoise  = signal_input[1][4][0]
        self.SNR       = signal_input[1][4][1]   
        self.singleInv = signal_input[1][5]                   # RELEVANT FOR JI
        self.jointInv  = signal_input[1][6]                   # RELEVANT FOR JI
        self.signType  = signal_input[1][7]
        self.signal_cc = sum(signal_input[1][:3])             # RELEVANT FOR JI
        
        self.noParam   = self.T2.GAUSS.noPara
        self.noIter    = self.PSO.noIter
        self.noPart    = self.PSO.noPart    
        self.noPSOIter = noPSOIter

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
        self.modV0     = {'T1':[],'T2':[],'T2S':[]}
        self.modV1     = {'T1':[],'T2':[],'T2S':[]}
        self.vel       = {'T1':[],'T2':[],'T2S':[]}
        self.fit       = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        self.synDat    = {'T1':[],'T2':[],'T2S':[]}
        self.noSteps   = {'T1':  self.Inversion.datSpaceT1,
                          'T2':  self.Inversion.datSpaceT2,
                          'T2S': self.Inversion.datSpaceT2S}

        # Filling dictionaries with values: should be implemented as function
        for signal in self.signType:
            
            if self.modParam[0] == 'class' and self.invOpt == 'V0':
                self.__buildModelVector_V0_PARAM__(signal)
            if self.modParam[0] == 'json' and self.invOpt == 'V0':
                self.__buildModelVector_V0_JSON__(signal)
            
            self.mod             = copy.deepcopy(self.modV0)
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
        
###############################################################################
###############################################################################
###############################################################################  
      
    def __buildModelVector_V0_PARAM__(self, signal):
        
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
        self.modV0[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
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

            self.modV0[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
                                                   self.m2[signal],   self.m2_sig[signal],
                                                   self.m3[signal],   self.m3_sig[signal],
                                                   self.int2[signal], self.int3[signal],
                                                   self.MWF[signal]))

###############################################################################
     
    def __buildModelVector_V0_JSON__(self, signal):
        
        m1     = self.modParam[2][f'SynData{signal}']['m1']
        m1_sig = self.modParam[2][f'SynData{signal}']['m1_sig']
        m2     = self.modParam[2][f'SynData{signal}']['m2']
        m2_sig = self.modParam[2][f'SynData{signal}']['m2_sig']
        int2   = self.modParam[2][f'SynData{signal}']['integ2']
        MWF    = self.modParam[2][f'SynData{signal}']['MWF']

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
        self.modV0[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
                                               self.m2[signal],   self.m2_sig[signal], 
                                               self.int2[signal], self.MWF[signal]))
        
        if self.noPeaks == 'DIRAC':
            m3     = self.modParam[2][f'SynData{signal}']['m3']
            m3_sig = self.modParam[2][f'SynData{signal}']['m3_sig']
            int3   = self.modParam[2][f'SynData{signal}']['integ3']

            self.noParam        = self.T2.DIRAC.noPara
            self.m3[signal]     = np.random.uniform(m3[0], m3[1], self.noPart)
            self.m3_sig[signal] = np.random.uniform(m3_sig[0], m3_sig[1], self.noPart)
            self.int3[signal]   = np.random.uniform(int3[0], int3[1], self.noPart)

            self.modV0[signal]  = np.column_stack((self.m1[signal],   self.m1_sig[signal], 
                                                   self.m2[signal],   self.m2_sig[signal],
                                                   self.m3[signal],   self.m3_sig[signal],
                                                   self.int2[signal], self.int3[signal],
                                                   self.MWF[signal]))
        
###############################################################################

    def getObsDataFromMeas(self, position, **kwargs):
        
        '''    
        Creates a dictionary with invivo MRI data. Observed signal is
        will be smothed by a gauss-filtered and normalized to 1.  

        Three MRI relaxation signals can be inverted single or joint:
            sig: T1, T2, T2S            

        Input parameters
            position: (y-coord, x-coord, slice)\n
            pathSig: signal file if single inversion\n
            pathT1,pathT2 and/or pathT2S: signal files for joint inversion

        Returns: 
            observed MRI relaxation data (type: dict of np.ndarray(s))
        
        '''

        yy, xx, noSlice = position[0], position[1], position[2]
        
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

    def createSynData_fast(self):

        '''    
        Creates synthetic MRI decay signals for a number of particles.

        Three MRI relaxation signals can be created: T1, T2, T2S.  
            
        The method is matrix-based on integration over 
        two (noPeaks: GAUSS) or three peaks (noPeaks:DIRAC).
        
        Returns:
            Dictionary .synDat with np.ndarrays in the shape [noPart, signal length]. 
        
        '''
                    
        if self.noPeaks == 'GAUSS':
                        
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


        if self.noPeaks == 'DIRAC':

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
            invOpt [V0, V1] - method for joint inversion
                              V0: one single model vector for each signal\n
                              V1: one common model vector for both signals
            
            lpNorm [l1, L2] - Lp-Norm for generalization
                              L1: linear or L2: quadratic
                              
        Construction work: L1-Norm for single inversion can be tested, beta.
                           V0 not fully implemented yet!
        '''
        
        if self.singleInv==True:
                
            sig = self.signType[0]
                
            for ii in range(0, self.noPart):
                
                if self.lpNorm == 'L1':
                    self.fit[sig][ii] = np.sum(np.abs(self.obsData[sig]-self.synDat[sig][ii]))/len(self.obsData[sig])

                if self.lpNorm == 'L2': # adjusted to Sego-Code: Frobenius Norm 
                    self.fit[sig][ii] = np.sqrt(np.sum((self.obsData[sig]-self.synDat[sig][ii])**2))/len(self.obsData[sig])


        if self.jointInv==True and self.invOpt=='V0':

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for ii in range(0, self.noPart):
                
                if self.lpNorm == 'L1':
                    sys.exit('ATTENTION: L1-Norm not yet implemeted for joint inversion.')
                    continue

                if self.lpNorm == 'L2':
                    term_sig_1      = np.sum((self.obsData[i]-self.synDat[i][ii,:])**2)
                    weight_alpha    = 1
                    term_sig_2      = np.sum((self.obsData[j]-self.synDat[j][ii,:])**2)
                    weight_beta     = 1
                    term_MWF        = (self.MWF[i][ii]-self.MWF[j][ii])**2
                    weight_gamma    = 1
                    
                    # if ii == 1:
                    #     print(self.obsData[j])
                    #     print(self.synDat[j][ii,:])
                    #     print(f'm1: {self.m1[j][ii]}, m1_sig: {self.m1_sig[j][ii]},'
                    #           f'm2: {self.m2[j][ii]}, m2_sig: {self.m2_sig[j][ii]},'
                    #           f'm3: {self.m3[j][ii]}, m3_sig: {self.m3_sig[j][ii]},'
                    #           f'int2: {self.int2[j][ii]}, int3: {self.int3[j][ii]}')
                    #     print(f'T2:  {term_sig_1}')
                    #     print(f'T2S: {term_sig_2}')
                    #     print(f'MWF: {term_MWF}\n')
                        
                        # if ii == 1:
                        #     sys.exit()
                    
                    self.fit[k][ii] = weight_alpha  * np.sqrt(term_sig_1)/len(self.obsData[i])  + \
                                      weight_beta   * np.sqrt(term_sig_2)/len(self.obsData[j])  + \
                                      weight_gamma  * term_MWF
        
        if self.jointInv==True and self.invOpt=='V1':
            
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
        
        '''        
        Return:
        '''
        if self.lpNorm == 'L1':
            return np.sum(np.abs(data1-data2))/len(data1)
    
        if self.lpNorm == 'L2':
            return np.sqrt(np.sum((data1-data2)**2)/len(data1))
            
###############################################################################            
    
    def bestLocal(self):

        '''
        Update of the best local position/parameters for each swarm particle.
        
        Parameters for use in PSO class initiation:
            invOpt [V0, V1] - method for joint inversion
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

        if self.jointInv==True and self.invOpt=='V0':   

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            
            for sig in self.signType:

                self.bestMWF[sig][fit[k]<=bfit[k]]    = MWF[sig][fit[k]    <= bfit[k]]
                self.bestMod[sig][fit[k]<=bfit[k]]    = mod[sig][fit[k]    <= bfit[k]]
                self.bestSynDat[sig][fit[k]<=bfit[k]] = synDat[sig][fit[k] <= bfit[k]]
                
            self.bestFit[k][fit[k]<=bfit[k]]          = fit[k][fit[k]    <= bfit[k]]
            self.bestMWF[k]                           = np.sum([self.bestMWF[i],self.bestMWF[j]], axis=0)/self.signal_cc

        if self.jointInv==True and self.invOpt=='V1':   

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
        
        if self.jointInv==True and self.invOpt=='V0':
            
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

        if self.jointInv==True and self.invOpt=='V1':
            
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

###############################################################################
    
    def write_JSON(self, savepath: str):
        
        signal      = ('').join([sig for sig in self.signType])
        savepath    = f'{savepath}{signal}_param.json'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
            
        if self.singleInv==True:
          
            datspace    = self.Inversion.datSpaceT2 if self.signType[0]=='T2' else self.Inversion.datSpaceT2S                
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
                                "_timepoints":           datspace},
                            f"SynData{self.signType[0]}":{
                                "_Solver": 'matrix-based',
                                "_Model":  self.noPeaks,
                                "_m1":     getattr(getattr(self, self.signType[0]), self.noPeaks).m1,
                                "_m1_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m1_sig, 
                                "_m2":     getattr(getattr(self, self.signType[0]), self.noPeaks).m2,
                                "_m2_sig": getattr(getattr(self, self.signType[0]), self.noPeaks).m2_sig, 
                                "_m3":     None,
                                "_m3_sig": None,
                                "_integ2": getattr(getattr(self, self.signType[0]), self.noPeaks).int2,
                                "_integ3": None,
                                "_MWF":    getattr(getattr(self, self.signType[0]), self.noPeaks).MWF
                            }}
            
            if self.noPeaks == 'DIRAC':
                jsonData[f"SynData{self.signType[0]}"]["_m3"]     = getattr(getattr(self, self.signType[0]), self.noPeaks).m3
                jsonData[f"SynData{self.signType[0]}"]["_m3_sig"] = getattr(getattr(self, self.signType[0]), self.noPeaks).m3_sig
                jsonData[f"SynData{self.signType[0]}"]["_integ3"] = getattr(getattr(self, self.signType[0]), self.noPeaks).int3
               
            if self.noPeaks == 'GAUSS':
                syn_data = jsonData[f"SynData{self.signType[0]}"]
                syn_data = {k: v for k, v in syn_data.items() if v is not None}
                jsonData[f"SynData{self.signType[0]}"] = syn_data
    
            f = json.dumps(jsonData).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                            
            with open(savepath, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
        if self.jointInv==True:
            
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
                                "_m3":     None,
                                "_m3_sig": None,
                                "_integ2": getattr(getattr(self, self.signType[0]), self.noPeaks).int2,
                                "_integ3": None,
                                "_MWF":    getattr(getattr(self, self.signType[0]), self.noPeaks).MWF},
                            "SynDataT2S":{
                                "_Solver": 'matrix-based',
                                "_Model":  self.noPeaks,
                                "_m1":     getattr(getattr(self, self.signType[1]), self.noPeaks).m1,
                                "_m1_sig": getattr(getattr(self, self.signType[1]), self.noPeaks).m1_sig, 
                                "_m2":     getattr(getattr(self, self.signType[1]), self.noPeaks).m2,
                                "_m2_sig": getattr(getattr(self, self.signType[1]), self.noPeaks).m2_sig, 
                                "_m3":     None,
                                "_m3_sig": None,
                                "_integ2": getattr(getattr(self, self.signType[1]), self.noPeaks).int2,
                                "_integ3": None,
                                "_MWF":    getattr(getattr(self, self.signType[1]), self.noPeaks).MWF
                            }}

            if self.noPeaks == 'DIRAC':
                for sig in self.signType:                    
                    jsonData[f"SynData{sig}"]["_m3"]     = getattr(getattr(self, sig), self.noPeaks).m3
                    jsonData[f"SynData{sig}"]["_m3_sig"] = getattr(getattr(self, sig), self.noPeaks).m3_sig
                    jsonData[f"SynData{sig}"]["_integ3"] = getattr(getattr(self, sig), self.noPeaks).int3
               
            if self.noPeaks == 'GAUSS':
                for sig in self.signType:  
                    syn_data = jsonData[f"SynData{sig}"]
                    syn_data = {k: v for k, v in syn_data.items() if v is not None}
                    jsonData[f"SynData{sig}"] = syn_data

            f = json.dumps(jsonData).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                        
            with open(savepath, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
###############################################################################
    
    def result2array(self, results:dict, result_map:np.array, kk:int, **kwargs):
        
        cutThresh      = kwargs.get('cutThresh', (None,None,False))
        cutMask        = kwargs.get('cutMask', (None, False))
        calcBestResult = kwargs.get('calcBestResult', False)
        arrayType      = kwargs.get('arrayType', 'Slice')
        
        results_ = [i for i in results if i != None]
        
        if arrayType == 'Pixel':
            
            for sig in self.signType:
                
                for i, line in enumerate(results_):
                    result_map[sig][:-1, i] = line[f'mod{sig}']
                    result_map[sig][-1,  i] = line['fit']
        
        if arrayType == 'Slice':
            
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
                result_map[j,:,:,i][np.isnan(mask)]=np.nan
        
        return result_map
            
###############################################################################    
        
    def initPS(self):
        
        # solving the forward modelling for each particle
        self.createSynData_fast()
        
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
    
    def execPSO(self, obsData: np.array, plotIterTest=False):

        self.plotResult = plotIterTest
        self.obsData    = obsData
        
        self.initPS()
        
        for jj in range(self.noIter)[:]:

            # update particle position in the global search space
            # each Particle  -->  locVector = w*v1 + c1*r1*(m[i+1]-m[i])              
            self.updatePos()
            
            # # check the limits of new model values
            # # --> using better np.clip() funtion instead of generic expressions            
            self.checkLim()
    
            # update synthetic data set
            self.createSynData_fast()
                        
            # update particle fitness
            self.fitness()
            
            # update local best for each particle
            self.bestLocal()
      
            # update global best for the particle swarm
            self.bestGlobal()

            # collect best global fit and indicies for a number of n iterations
            if self.plotResult==True and self.singleInv==True:
                sig = self.signType[0]
                self.globInd_list[sig].append(self.globIndex[sig]+1)
                self.globFit_list[sig].append(self.globFit[sig])
                self.globMWF_list[sig].append(self.globMod[sig][-1])

###############################################################################

    def log(self, startTime, string='', dim='sek', boolean=False):
        
        '''
        Parameters
        ----------
        startTime : time.time() object

        string : string

        dim : output dimension
              ms  - milli seconds
              mus - micro seconds
              MS  - min:sec (default)
              HMS - hr:min:sec.

        Returns : None
        '''

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