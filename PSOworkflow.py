# -*- coding: utf-8 -*-
"""
Class and methods for applying particle swarm optimizing (PSO) on invivo MRI data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)
"""

import time, os, copy, json
import numpy             as     np
from   numba             import njit
import matplotlib.pyplot as     plt

from   functools         import partial
from   scipy.ndimage     import gaussian_filter, label

import helpTools         as     hlp
from   PSOparameters     import Parameters   as PM


# jit kompilationen für schnellere performance und lesbarkeit
@njit#(fastmath=True, cache=True, boundscheck=False)
def compute_gaussian_fast(x_sub, values, fac_sig_sqrt, fac_pre):
    _dx  = x_sub - values
    _exp = np.exp(-np.square(_dx) * fac_sig_sqrt)  
    return fac_pre * _exp

@njit#(fastmath=True, cache=True, boundscheck=False)
def compute_matmul_fast(gauss, sys_mat):
    return gauss @ sys_mat

@njit#(fastmath=True, cache=True, boundscheck=False)
def compute_CT2S_fast(MW_f, FW_f, phi, mat1_array, mat2_array, mult_CT2S):
                        
    exp_MW  = np.exp(mult_CT2S * MW_f)
    exp_FW  = np.exp(mult_CT2S * FW_f)
    exp_phi = np.exp(-1j * phi)

    return (mat1_array * exp_MW + mat2_array * exp_FW) * exp_phi


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
        
    def __init__(self, config_data: dict, **kwargs):

        # get optional kwargs if model
        self.plot_mod  = kwargs.get('model_plot', (False, ''))[0]
        self.save_path = kwargs.get('model_plot', (False, ''))[1]
        self.init_mat  = kwargs.get('init_matrix', False)
        self.sys_param = kwargs.get('sys_param', dict())
        self.position  = kwargs.get('position', (0,0))
        self.constants = kwargs.get('constants', dict())
        
        # config to class object
        self._config_to_object(config_data=config_data)
        
        # Inherit parameters and methods from:
        # (a) class containing parameter for
        #     (a.a) generating obsData and synData of T1, T2, T2*
        #     (a.b) performing PSO algorithm and cycles
        # (b) forward solver class inversion.Gaussian
        #     (b.a) methods for generating obsData T1, T2, T2* via integration
        super().__init__()

        # possible keyword arguments, e.g. randseed

        self.randSeedBool  = self.config.PSO_spec.dyn.rand[0]
        self.randSeedValue = self.config.PSO_spec.dyn.rand[1]
        if self.randSeedBool == True:
            np.random.seed(self.randSeedValue)
            
        self.lpNorm     = self.config.PSO_spec.lp_norm
        self.invOpt     = self.config.PSO_spec.mod_vec
        self.noPeaks    = self.config.PSO_spec.peaks
        self.modParam   = kwargs.get('modParam', ('class', None, None))
        self.use_njit   = self.config.PSO_spec.math.njit
        self.norm_max   = self.config.PSO_spec.math.norm
        
        # get paremeters from the configuration file
        self.invT1      = self.config.source.signal.T1
        self.invT2      = self.config.source.signal.T2
        self.invT2S     = self.config.source.signal.T2S
        self.invCT2S    = self.config.source.signal.CT2S
        self.dataSource = self.config.source.data.obs_data
        self.addNoise   = self.config.source.data.add_noise[0]
        self.SNR        = self.config.source.data.add_noise[1]   
        self.singleInv  = self.config.source.signal.SI
        self.jointInv   = self.config.source.signal.JI
        self.signType   = [sig for sig, inv in zip(['T1','T2','T2S'], 
                          [self.invT1, self.invT2, self.invT2S]) if inv]
        
        self.noIter     = self.config.PSO_spec.iter
        self.noPart     = self.config.PSO_spec.part
        self.noPSOIter  = self.config.PSO_spec.PSO_iter.slice

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
        self.MW_f      = {'T2S': []} # myelin water         - frequency shift
        self.FW_f      = {'T2S': []} # free water           - frequency shift
        self.AW_f      = {'T2S': []} # axonal water         - frequency shift
        self.EW_f      = {'T2S': []} # extracelular water   - frequency shift
        self.phi       = {'T2S': []} # global phase shift
        
        # PSO parameters
        self.noParam   = {'T1':[],'T2':[],'T2S':[]}
        self.modV0     = {'T1':[],'T2':[],'T2S':[]}
        self.modV1     = {'T1':[],'T2':[],'T2S':[]}
        self.vel       = {'T1':[],'T2':[],'T2S':[]}
        self.fit       = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        self.synDat    = {'T1':[],'T2':[],'T2S':[]}
        self.noSteps   = {'T1':   self.Inversion.datSpaceT1,
                          'T2':   self.Inversion.datSpaceT2,
                          'T2S':  self.Inversion.datSpaceT2S}

        # Filling dictionaries with values: should be implemented as function
        for signal in self.signType:
            
            if self.modParam[0] == 'class' and self.invOpt == 'V0':
                self.__buildModelVector_V0_PARAM__(signal)
            if self.modParam[0] == 'json' and self.invOpt == 'V0':
                self.__buildModelVector_V0_JSON__(signal)
            
            self.mod             = copy.deepcopy(self.modV0)
            self.vel[signal]     = np.random.uniform(-0.3, 0.3, (self.noPart, self.noParam[signal]))*self.mod[signal]      
            self.fit[signal]     = np.empty(self.noPart)
            
            if self.invCT2S == True:
                self.synDat[signal] = np.empty((self.noPart, self.noSteps[signal]), dtype=np.complex128)
            else:  # T1, T2, T2S
                self.synDat[signal] = np.empty((self.noPart, self.noSteps[signal]), dtype=np.float64)
        
        # Continue Filling dictionaries with values if joint inversion was chosen
        if self.jointInv==True:
            self.fit[self.sigJoint] = np.empty(self.noPart)
            self.MWF[self.sigJoint] = np.empty(self.noPart)
            
        # parameter settings for a best local fit        
        self.bestMWF    = {key: np.full_like(self.MWF[key],    np.inf) for key in self.MWF.keys()}
        self.bestFit    = {key: np.full_like(self.fit[key],    np.inf) for key in self.fit.keys()}
        self.bestMod    = {key: np.full_like(self.mod[key],    np.inf) for key in self.mod.keys()}
        self.bestSynDat = {key: np.full_like(self.synDat[key], np.inf) for key in self.synDat.keys()}
        
        # parameter settings for a best global fit              
        self.globFit    = {'T1':np.inf,'T2':np.inf,'T2S':np.inf,self.sigJoint:np.inf}
        self.globMod    = {'T1':[],'T2':[],'T2S':[]}
        self.globSynDat = {'T1':[],'T2':[],'T2S':[]}
        self.globIndex  = {'T1':[],'T2':[],'T2S':[],self.sigJoint:[]}
        
        # constant factors calculation
        if self.constants and self.init_mat == True:
          
            # forward solver (fs) constant factors
            self.fs_divisor_T2  = self.constants['fs_divisor_T2']
            self.fs_divisor_T2S = self.constants['fs_divisor_T2S']
            
            self.init_sysMatrix(rootMWF=self.sys_param, position=self.position) #, data_type=self.dataSource)
            
            if self.invT2 == True:
                self.fs_sysMatr_T2  = np.ascontiguousarray(self.sysMatrix['T2'].T*self.fs_divisor_T2)
            if self.invT2S == True:
                self.fs_sysMatr_T2S = np.ascontiguousarray(self.sysMatrix['T2S'].T*self.fs_divisor_T2S)
            if self.invCT2S == True:
                self.mult_CT2S   = self.constants['mult_CT2S']
                self.CT2S_TE     = self.sys_param['CT2S_TE']
            
            self.size_batches    = self.constants['size_batches']
            self.num_batches     = self.constants['num_batches']
            self.idx_slices      = self.constants['idx_slices']
            self.width_gauss     = self.constants['width_gauss']
            self.factor_gauss    = self.constants['factor_gauss']
            self.array_gauss_m1  = self.constants['array_gauss_m1']
            self.array_gauss_m2  = self.constants['array_gauss_m2']
            self.array_gauss_m3  = self.constants['array_gauss_m3']
            self.sysGrid['T2']   = self.constants['sys_grid_T2']
            self.sysGrid['T2S']  = self.constants['sys_grid_T2S']

        dummy = np.linspace(self.Inversion.T2min, self.Inversion.T2max, self.Inversion.modSpace)
        self.gauss_full_partial  = partial(hlp.gauss_full, x=dummy[None,:])

###############################################################################
###############################################################################
###############################################################################

    def __del__(self):
        self._close()
    
    def _close(self):
        self.__dict__.clear()

###############################################################################
###############################################################################
###############################################################################

    def _config_to_object(self, config_data: dict):
        
        '''
        Transfers the parameters from the configuration file into class internal objects.
        
        *args:
            config_data: content of the project configuration file
        '''
        
        self.config = type('Configuration', (), {})()
        
        def dict_to_attr(config_data, parent=None):
            if parent is None:
                parent = self.config
            
            for key, value in config_data.items():
                
                if isinstance(value, dict):
                    sub_obj = type(key, (), {})()
                    setattr(parent, str(key), sub_obj)
                    dict_to_attr(value, sub_obj)
                else:
                    setattr(parent, str(key), value)
        
        dict_to_attr(config_data)

###############################################################################
###############################################################################
###############################################################################

    def _config_to_attribute(self, signal):
        
        ''' Stores start model parameters for a signal into a PSO class object.
        
        *args:
            signal: MRI signal 
        '''
        
        obj      = getattr(self, signal)
        self.att = getattr(obj, self.noPeaks)        

###############################################################################
###############################################################################
###############################################################################
    
    def _constant_PSO_objects_test(self, CT2S_TE):
        
        ''' 
        Calculates constant values and return them as a dictionary object.
        '''
   
        const = {
                "fs_divisor_T2":  (self.Inversion.T2max - self.Inversion.T2min) / self.Inversion.modSpace,
                "fs_divisor_T2S": (self.Inversion.T2Smax - self.Inversion.T2Smin) / self.Inversion.modSpace,
                "size_batches":   self.config.PSO_spec.math.batch,
                "num_batches":    self.noPart // self.config.PSO_spec.math.batch,
                "idx_slices":     [slice(i*self.config.PSO_spec.math.batch, (i+1)*self.config.PSO_spec.math.batch) 
                                   for i in range(self.noPart // self.config.PSO_spec.math.batch)],
                "width_gauss":    self.config.PSO_spec.math.width,
                "factor_gauss":   np.sqrt(2*np.pi),
                "array_gauss_m1": np.zeros((self.config.PSO_spec.math.batch, self.Inversion.modSpace)),
                "array_gauss_m2": np.zeros((self.config.PSO_spec.math.batch, self.Inversion.modSpace)),
                "array_gauss_m3": np.zeros((self.config.PSO_spec.math.batch, self.Inversion.modSpace)),
                "sys_grid_T2":    np.linspace(self.Inversion.T2min, self.Inversion.T2max, self.Inversion.modSpace),
                "sys_grid_T2S":   np.linspace(self.Inversion.T2Smin, self.Inversion.T2Smax, self.Inversion.modSpace),
                }

        if self.invCT2S:
            const["mult_CT2S"] = -2j * np.pi * CT2S_TE / 1000
    
        return const

    def _constant_PSO_objects(self):
        
        ''' 
        Calculating constant values and store them in as PSO class objects.
        '''
        
        # forward solver (fs) constant factors
        self.fs_divisor_T2  = (self.Inversion.T2max  - self.Inversion.T2min)  / self.Inversion.modSpace
        self.fs_divisor_T2S = (self.Inversion.T2Smax - self.Inversion.T2Smin) / self.Inversion.modSpace

        self.fs_sysMatr_T2  = np.ascontiguousarray(self.sysMatrix['T2'].T*self.fs_divisor_T2)
        self.fs_sysMatr_T2S = np.ascontiguousarray(self.sysMatrix['T2S'].T*self.fs_divisor_T2S)
        # self.fs_sysMatr_T2  = self.sysMatrix['T2'].T * self.fs_divisor_T2
        # self.fs_sysMatr_T2S = self.sysMatrix['T2S'].T * self.fs_divisor_T2S
        
        self.size_batches   = self.config.PSO_spec.math.batch
        self.num_batches    = self.noPart // self.size_batches
        self.idx_slices     = [slice(i*self.size_batches, (i+1)*self.size_batches) for i in range(self.num_batches)]

        self.width_gauss    = self.config.PSO_spec.math.width
        self.factor_gauss   = np.sqrt(2*np.pi)
        self.array_gauss_m1 = np.zeros((self.size_batches, self.Inversion.modSpace))
        self.array_gauss_m2 = np.zeros((self.size_batches, self.Inversion.modSpace))
        self.array_gauss_m3 = np.zeros((self.size_batches, self.Inversion.modSpace))
        
###############################################################################
###############################################################################
###############################################################################  
      
    def __buildModelVector_V0_PARAM__(self, signal):
        
        ''' 
        Creates a model vector for particle swarm optimization. 
            Values are taken from the PSOparameters class.
        
        *args:
            signal: MRI signal 
        '''

        # # modelvector-matrix in the shape 3xMxN --> 3 signals, M particles, N parameters
        # #    mod = [[[mod11][mod12][.....][mod1N]]
        # #           [[mod21][mod12][.....][mod1N]]
        # #           [[.....][.....][.....][.....]]
        # #           [[modM1][modM2][.....][modMN]]]
        
        self._config_to_attribute(signal)

        self.m1[signal]     = np.random.uniform(self.att.m1[0],     self.att.m1[1],     self.noPart)
        self.m1_sig[signal] = np.random.uniform(self.att.m1_sig[0], self.att.m1_sig[1], self.noPart)
        self.m2[signal]     = np.random.uniform(self.att.m2[0],     self.att.m2[1],     self.noPart)
        self.m2_sig[signal] = np.random.uniform(self.att.m2_sig[0], self.att.m2_sig[1], self.noPart)
        self.int2[signal]   = np.random.uniform(self.att.int2[0],   self.att.int2[1],   self.noPart)
        self.MWF[signal]    = np.random.uniform(self.att.MWF[0],    self.att.MWF[1],    self.noPart)

        self.modV0[signal]  = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                               self.m2_sig[signal], self.int2[signal],   self.MWF[signal]))
                
        if self.invCT2S == True and signal == 'T2S' and self.noPeaks == 'GAUSS':
            
            self.MW_f[signal]  = np.random.uniform(self.att.MW_f[0], self.att.MW_f[1], self.noPart)
            self.FW_f[signal]  = np.random.uniform(self.att.FW_f[0], self.att.FW_f[1], self.noPart)
            self.phi[signal]   = np.random.uniform(self.att.phi[0],  self.att.phi[1],  self.noPart)
            
            self.modV0[signal] = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                  self.m2_sig[signal], self.int2[signal],   self.MWF[signal],
                                                  self.MW_f[signal],   self.FW_f[signal],   self.phi[signal]))
        
        if self.noPeaks == 'DIRAC':

            self.m3[signal]     = np.random.uniform(self.att.m3[0],     self.att.m3[1],      self.noPart)
            self.m3_sig[signal] = np.random.uniform(self.att.m3_sig[0], self.att.m3_sig[1],  self.noPart)
            self.int3[signal]   = np.random.uniform(self.att.int3[0],   self.att.int3[1],    self.noPart)

            self.modV0[signal]  = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                   self.m2_sig[signal], self.m3[signal],     self.m3_sig[signal],
                                                   self.int2[signal],   self.int3[signal],   self.MWF[signal]))
        
        if self.invCT2S == True and signal == 'T2S' and self.noPeaks == 'DIRAC':
            
            self.MW_f[signal]  = np.random.uniform(self.att.MW_f[0], self.att.MW_f[1], self.noPart)
            self.EW_f[signal]  = np.random.uniform(self.att.EW_f[0], self.att.EW_f[1], self.noPart)
            self.AW_f[signal]  = np.random.uniform(self.att.AW_f[0], self.att.AW_f[1], self.noPart)
            self.phi[signal]   = np.random.uniform(self.att.phi[0],  self.att.phi[1],  self.noPart)
            
            self.modV0[signal] = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                  self.m2_sig[signal], self.m3[signal],     self.m3_sig[signal],
                                                  self.int2[signal],   self.int3[signal],   self.MWF[signal],  
                                                  self.MW_f[signal],   self.EW_f[signal],   self.AW_f[signal], self.phi[signal]))

        self.noParam[signal] = self.modV0[signal].shape[-1]
        
###############################################################################
# wurde lange nicht angepasst! u.a.für joint inversion

    def __buildModelVector_V0_JSON__(self, signal):
        
        ''' 
        Creates a model vector for particle swarm optimization. 
            Values are taken from a dictionary with JSON file input parameters.
        
        *args:
            signal: MRI signal 
        '''
        
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

    def get_data_per_pixel(self, data_dir, position, filt_gauss=(0,0,0), norm_max=True, **kwargs):
        
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

        yy, xx, _slice = position[0], position[1], position[2]
        
        pathT1         = kwargs.get('pathT1',   None)
        pathT2         = kwargs.get('pathT2',   None)
        pathT2S        = kwargs.get('pathT2S',  None)
        pathT2SP       = kwargs.get('pathT2SP', None)
        pathTE         = kwargs.get('pathTE',   None)

        obs_data = {'T1':[],'T2':[],'T2S':[],'CT2S':[],'TE':[]}
        
        if self.invT1==True:
            img_nifti       = hlp.load_data(data_dir, pathT1)[0]
            raw_data        = img_nifti.get_fdata()
            dat_help        = gaussian_filter(raw_data[:,:,_slice], filt_gauss)
            obs_data['T1']  = dat_help[yy,xx]
            
            if norm_max == True:
                obs_data['T1']  = obs_data['T1']/np.max(obs_data['T1'])
            
        if self.invT2==True:
            img_nifti       = hlp.load_data(data_dir, pathT2)[0]
            raw_data        = img_nifti.get_fdata()
            dat_help        = gaussian_filter(raw_data[:,:,_slice], filt_gauss)
            obs_data['T2']  = dat_help[yy,xx]
            
            if norm_max == True:
                obs_data['T2']  = obs_data['T2']/np.max(obs_data['T2'])
            
        if self.invT2S==True and self.invCT2S==False:
            img_nifti       = hlp.load_data(data_dir, pathT2S)[0]
            raw_data        = img_nifti.get_fdata()
            dat_help        = gaussian_filter(raw_data[:,:,_slice], filt_gauss)                
            obs_data['T2S'] = dat_help[yy,xx]
            
            if norm_max == True:
                obs_data['T2S'] = obs_data['T2S']/np.max(obs_data['T2S']) 
            
        if self.invT2S==True and self.invCT2S==True:
            img_nifti_T2S  = hlp.load_data(data_dir, pathT2S)[0]
            img_nifti_T2SP = hlp.load_data(data_dir, pathT2SP)[0]
            raw_data_T2S   = img_nifti_T2S.get_fdata()
            raw_data_T2SP  = img_nifti_T2SP.get_fdata()
            
            factor = 1.0                          # phase values in radiants
            if np.max(raw_data_T2SP)>1000.0:      # assume unprocessed phase from dicom => -4096 < phase < 4096
                factor       = np.pi/4096.0
                complex_data = raw_data_T2S[:,:,_slice]*np.exp(1j*raw_data_T2SP[:,:,_slice]*factor)

            filt_real        = gaussian_filter(np.real(complex_data),filt_gauss)  # np.angle takes the phase
            filt_imag        = gaussian_filter(np.imag(complex_data),filt_gauss)  # np.abs takes the magnitude
                                
            obs_data['T2S']  = gaussian_filter(np.abs(complex_data), (1.0, 1.0, 0))[yy,xx]
            obs_data['T2S']  = obs_data['T2S']/np.max(obs_data['T2S']) 
            obs_data['CT2S'] = (filt_real + 1j*filt_imag)[yy,xx]
            obs_data['TE']   = np.load(os.path.join(data_dir, pathTE))

        return obs_data

###############################################################################

    def gauss_part(self, x, sigma, mean, scale, width=5):
        
        '''
        Creates the very same object as gauss_full(), but uses only maximum peak-intervals.
        
        args:
            x - system grid - and array which ...
        '''
        #######################################################################################
        ## approximating bell-curve width using sigma-width is best/fastest solution for now
        
        left  = np.searchsorted(x[0], np.min(mean) - width * np.max(sigma), side='left')
        right = np.searchsorted(x[0], np.max(mean) + width * np.max(sigma), side='right')
        x_sub = x[0, left:right]
    
        exp_arg = -np.square(x_sub[None,:] - mean) / (2 * sigma**2)
        
        array                = np.zeros((sigma.shape[0], x.shape[1]))
        array[:, left:right] = scale / (np.sqrt(2 * np.pi) * sigma) * np.exp(exp_arg)
    
        return array
        
    def compute_constants_gauss(self, widths, weights):
        
        _sigma_square_inv = 1 / (2 * widths**2)
        _prefactor        = weights / (self.factor_gauss * widths)
        
        return _sigma_square_inv, _prefactor
 
    def compute_intervall_gauss(self, widths, values, scale, signal='T2'):
        
        max_sig   = widths.reshape(self.num_batches, self.size_batches).max(axis=1)
        max_scale = scale.reshape(self.num_batches, self.size_batches).max(axis=1)
      
        with np.errstate(divide='ignore', invalid='ignore'):
            inverse1 = 1e-3 * np.sqrt(2 * np.pi) * max_sig / max_scale
            inverse2 = np.sqrt(-2 * max_sig**2 * np.log(inverse1))
            inverse2 = np.where(np.isfinite(inverse2), inverse2, 0)
        
        _left     = values.reshape(self.num_batches, self.size_batches).min(axis=1) - inverse2
        _right    = values.reshape(self.num_batches, self.size_batches).max(axis=1) + inverse2
        
        left    = np.searchsorted(self.sysGrid[signal], _left, side='left')   - 1
        right   = np.searchsorted(self.sysGrid[signal], _right, side='right') + 1
        
        left[left < 0] = 0
    
        return left, right         
    
    def compute_gaussian(self, sigma, values, array, fac_sig_sqrt, fac_pre, signal='T2', partial=(False,None,None)):
        
        if partial[0] == True:
            
            sl          = slice(partial[1], partial[2])            
            x_sub       = self.sysGrid[signal][None, sl]

            _dx         = x_sub - values                         # = x - mean
            _exp        = np.exp(-np.square(_dx) * fac_sig_sqrt) # fac_sig_sqrt = 1 / (2 * sigma ** 2))

            return fac_pre * _exp
        
        else:            
            _dx         = self.sysGrid['T2'][None, :] - values            
            return fac_pre * np.exp(-np.square(_dx) * fac_sig_sqrt)
    

    def createSynData_fast(self, ind_iter: int):

        '''    
        Creates synthetic MRI decay signals for a number of particles.

        Three MRI relaxation signals can be created: T1, T2, T2S.  
            
        The method is matrix-based on integration over 
        two (noPeaks: GAUSS) or three peaks (noPeaks:DIRAC).
        
        Returns:
            Dictionary .synDat with np.ndarrays in the shape [noPart, signal length]. 
        
        '''

        if self.noPeaks == 'GAUSS':
                        
            if self.invT2==True:
                
                values    = np.array([self.m1['T2'],self.m2['T2']])
                weights   = np.array([self.int2['T2']*self.MWF['T2']/(1-self.MWF['T2']),self.int2['T2']])
                widths    = np.array([self.m1_sig['T2'],self.m2_sig['T2']])                
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2 = {}, {}
                
                width_max = np.max(widths, axis=1)
                
                sig_sq, fac_pre   = self.compute_constants_gauss(widths, weights)              
                
                left_m1, right_m1 = self.compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2')
                left_m2, right_m2 = self.compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2')

                for i, idx in enumerate(self.idx_slices):
                    
                    if self.use_njit == True:

                        m1 = compute_gaussian_fast(self.sysGrid['T2'][None, left_m1[i]:right_m1[i]], 
                                                   values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = compute_gaussian_fast(self.sysGrid['T2'][None, left_m2[i]:right_m2[i]], 
                                                   values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])

                        mat_m1[i] = compute_matmul_fast(m1, self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = compute_matmul_fast(m2, self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :])
                                    
                    else:                   
                        if width_max[0] <= 24:
                            m1 = self.compute_gaussian(widths[0,idx,None], values[0,idx,None], self.array_gauss_m1, sig_sq[0,idx,None], 
                                                       fac_pre[0,idx,None], partial=[True,left_m1[i],right_m1[i]])                            
                            mat_m1[i] = m1 @ self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :]
                        else:
                            m1 = self.gauss_full_partial(sigma=widths[0, idx, None], mean=values[0, idx, None], scale=weights[0, idx, None])
    
                        if width_max[1] <= 21:
                            m2 = self.compute_gaussian(widths[1,idx,None], values[1,idx,None], self.array_gauss_m2, sig_sq[1,idx,None], 
                                                       fac_pre[1,idx,None], partial=[True,left_m2[i],right_m2[i]])
                            mat_m2[i] = m2 @ self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :]
                        else:
                            m2 = self.gauss_full_partial(sigma=widths[1, idx, None], mean=values[1, idx, None], scale=weights[1, idx, None])
                
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                
                self.synDat['T2'] = mat1_array + mat2_array
                
                if self.norm_max == True:
                    max_val = np.max(self.synDat['T2'], axis=1, keepdims=True)
                    self.synDat['T2'] = self.synDat['T2']/max_val
                

            if self.invT2S==True:

                values    = np.array([self.m1['T2S'],self.m2['T2S']])
                weights   = np.array([self.int2['T2S']*self.MWF['T2S']/(1-self.MWF['T2S']),self.int2['T2S']])
                widths    = np.array([self.m1_sig['T2S'],self.m2_sig['T2S']])        
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2 = {}, {}
                
                width_max = np.max(widths, axis=1)
                
                sig_sq, fac_pre   = self.compute_constants_gauss(widths, weights)              
                
                left_m1, right_m1 = self.compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2S')
                left_m2, right_m2 = self.compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2S')
                                
                for i, idx in enumerate(self.idx_slices):
                    
                    if self.use_njit == True:
                        
                        m1 = compute_gaussian_fast(self.sysGrid['T2S'][None, left_m1[i]:right_m1[i]], 
                                                   values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = compute_gaussian_fast(self.sysGrid['T2S'][None, left_m2[i]:right_m2[i]], 
                                                   values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                        
                        mat_m1[i] = compute_matmul_fast(m1, self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = compute_matmul_fast(m2, self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :])
                                                     
                    else:
                        if width_max[0] <= 24:
                            m1 = self.compute_gaussian(widths[0,idx,None], values[0,idx,None], self.array_gauss_m1, sig_sq[0,idx,None], 
                                                       fac_pre[0,idx,None], partial=[True,left_m1[i],right_m1[i]])
                            mat_m1[i] = m1 @ self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :]
                        else:
                            m1 = self.gauss_full_partial(sigma=widths[0, idx, None], mean=values[0, idx, None], scale=weights[0, idx, None])
    
                        if width_max[1] <= 21:
                            m2 = self.compute_gaussian(widths[1,idx,None], values[1,idx,None], self.array_gauss_m2, sig_sq[1,idx,None], 
                                                       fac_pre[1,idx,None], partial=[True,left_m2[i],right_m2[i]])
                            mat_m2[i] = m2 @ self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :]
                        else:
                            m2 = self.gauss_full_partial(sigma=widths[1, idx, None], mean=values[1, idx, None], scale=weights[1, idx, None])                    

                ## version 2.0.0 alpha
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                    
                if self.invCT2S==False:
                    self.synDat['T2S'] = mat1_array + mat2_array
                    
                    if self.norm_max == True:
                        max_val = np.max(self.synDat['T2S'], axis=1, keepdims=True)
                        self.synDat['T2S'] = self.synDat['T2S']/max_val

                
                if self.invCT2S:

                    if self.use_njit == True:                        
                        self.synDat['T2S'] = compute_CT2S_fast(self.MW_f['T2S'][:, None], self.FW_f['T2S'][:, None], self.phi['T2S'][:, None], 
                                                               mat1_array, mat2_array, self.mult_CT2S)
                    
                    else:                        
                        exp_MW  = np.exp(self.mult_CT2S * self.MW_f['T2S'][:, None])
                        exp_FW  = np.exp(self.mult_CT2S * self.FW_f['T2S'][:, None])
                        exp_phi = np.exp(-1j * self.phi['T2S'][:, None])
                    
                        self.synDat['T2S']  = (mat1_array * exp_MW + mat2_array * exp_FW) * exp_phi
                
                    if self.norm_max == True:
                        max_val = np.max(np.abs(self.synDat['T2S']), axis=1, keepdims=True)
                        max_val[max_val == 0] = 1
                        self.synDat['T2S'] = self.synDat['T2S'] / max_val
                            

        if self.noPeaks == 'DIRAC':             
            
            if self.invT2==True:
                
                values  = np.array([self.m1['T2'], self.m2['T2'], self.m3['T2']])
                weights = np.array([(self.int2['T2']+self.int3['T2'])*self.MWF['T2']/(1 - self.MWF['T2']), self.int2['T2'], self.int3['T2']])
                widths  = np.array([self.m1_sig['T2'], self.m2_sig['T2'], self.m3_sig['T2']])
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values     = np.take_along_axis(values,  idx_sort, axis=1)
                weights    = np.take_along_axis(weights, idx_sort, axis=1)
                widths     = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2, mat_m3 = {}, {}, {}
                                
                width_max = np.max(widths, axis=1)
                
                sig_sq, fac_pre   = self.compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self.compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2')
                left_m2, right_m2 = self.compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2')
                left_m3, right_m3 = self.compute_intervall_gauss(widths[2], values[2], weights[2], signal='T2')          

                for i, idx in enumerate(self.idx_slices):        

                    if self.use_njit == True:                        
                        m1 = compute_gaussian_fast(self.sysGrid['T2'][None, left_m1[i]:right_m1[i]], 
                                                   values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = compute_gaussian_fast(self.sysGrid['T2'][None, left_m2[i]:right_m2[i]], 
                                                   values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                        m3 = compute_gaussian_fast(self.sysGrid['T2'][None, left_m3[i]:right_m3[i]], 
                                                   values[2, idx, None], sig_sq[2, idx, None], fac_pre[2, idx, None])
                        
                        mat_m1[i] = compute_matmul_fast(m1, self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = compute_matmul_fast(m2, self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :])
                        mat_m3[i] = compute_matmul_fast(m3, self.fs_sysMatr_T2[left_m3[i]:right_m3[i], :])
                    else:                   
                        if width_max[0] != 24:
                            m1 = self.compute_gaussian(widths[0,idx,None], values[0,idx,None], self.array_gauss_m1, sig_sq[0,idx,None], 
                                                       fac_pre[0,idx,None], partial=[True,left_m1[i],right_m1[i]])
                            mat_m1[i] = m1 @ self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :]
                        else:
                            m1 = self.gauss_full_partial(sigma=widths[0, idx, None], mean=values[0, idx, None], scale=weights[0, idx, None])
    
                        if width_max[1] != 21:
                            m2 = self.compute_gaussian(widths[1,idx,None], values[1,idx,None], self.array_gauss_m2, sig_sq[1,idx,None], 
                                                       fac_pre[1,idx,None], partial=[True,left_m2[i],right_m2[i]])
                            mat_m2[i] = m2 @ self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :]
                        else:
                            m2 = self.gauss_full_partial(sigma=widths[1, idx, None], mean=values[1, idx, None], scale=weights[1, idx, None])
    
                        if width_max[2] != 21:
                            m3 = self.compute_gaussian(widths[2,idx,None], values[2,idx,None], self.array_gauss_m3, sig_sq[2,idx,None], 
                                                       fac_pre[2,idx,None], partial=[True,left_m3[i],right_m3[i]])
                            mat_m3[i] = m3 @ self.fs_sysMatr_T2[left_m3[i]:right_m3[i], :]
                        else:
                            m3 = self.gauss_full_partial(sigma=widths[2, idx, None], mean=values[2, idx, None], scale=weights[2, idx, None])

                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                mat3_array = np.concatenate([mat_m3[k] for k in mat_m3.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                mat3_array = mat3_array[idx_unsort[2], :]
                
                self.synDat['T2'] = mat1_array + mat2_array + mat3_array               
                                
                max_val = np.max(self.synDat['T2'], axis=1, keepdims=True)
                self.synDat['T2'] = self.synDat['T2']/max_val
                                                        
            if self.invT2S==True:
                
                values    = np.array([self.m1['T2S'],self.m2['T2S'],self.m3['T2S']])
                weights   = np.array([(self.int2['T2S']+self.int3['T2S'])*self.MWF['T2S']/(1-self.MWF['T2S']), self.int2['T2S'],self.int3['T2S']])
                widths    = np.array([self.m1_sig['T2S'],self.m2_sig['T2S'],self.m3_sig['T2S']])                         
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2, mat_m3 = {}, {}, {}
                                
                width_max = np.max(widths, axis=1)
                
                sig_sq, fac_pre   = self.compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self.compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2S')
                left_m2, right_m2 = self.compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2S')
                left_m3, right_m3 = self.compute_intervall_gauss(widths[2], values[2], weights[2], signal='T2S')   

                for i, idx in enumerate(self.idx_slices):
                   
                    if self.use_njit == True:                        
                        m1 = compute_gaussian_fast(self.sysGrid['T2S'][None, left_m1[i]:right_m1[i]], 
                                                   values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = compute_gaussian_fast(self.sysGrid['T2S'][None, left_m2[i]:right_m2[i]], 
                                                   values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                        m3 = compute_gaussian_fast(self.sysGrid['T2S'][None, left_m3[i]:right_m3[i]], 
                                                   values[2, idx, None], sig_sq[2, idx, None], fac_pre[2, idx, None])
                        
                        mat_m1[i] = compute_matmul_fast(m1, self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = compute_matmul_fast(m2, self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :])
                        mat_m3[i] = compute_matmul_fast(m3, self.fs_sysMatr_T2S[left_m3[i]:right_m3[i], :])
                    else:             
                        if width_max[0] != 24:
                            m1 = self.compute_gaussian(widths[0,idx,None], values[0,idx,None], self.array_gauss_m1, sig_sq[0,idx,None], 
                                                       fac_pre[0,idx,None], partial=[True,left_m1[i],right_m1[i]])
                            mat_m1[i] = m1 @ self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :]
                        else:
                            m1 = self.compute_gaussian(values[0,idx,None], self.array_gauss_m1, sig_sq[0,idx,None], fac_pre[0,idx,None])
    
                        if width_max[1] != 21:
                            m2 = self.compute_gaussian(widths[1,idx,None], values[1,idx,None], self.array_gauss_m2, sig_sq[1,idx,None], 
                                                       fac_pre[1,idx,None], partial=[True,left_m2[i],right_m2[i]])
                            mat_m2[i] = m2 @ self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :]
                        else:
                            m2 = self.compute_gaussian(values[1,idx,None], self.array_gauss_m2, sig_sq[1,idx,None], fac_pre[1,idx,None])
    
                        if width_max[2] != 21:
                            m3 = self.compute_gaussian(widths[2,idx,None], values[2,idx,None], self.array_gauss_m3, sig_sq[2,idx,None], 
                                                       fac_pre[2,idx,None], partial=[True,left_m3[i],right_m3[i]])
                            mat_m3[i] = m3 @ self.fs_sysMatr_T2S[left_m3[i]:right_m3[i], :]
                        else:
                            m3 = self.compute_gaussian(values[2,idx,None], self.array_gauss_m3, sig_sq[2,idx,None], fac_pre[2,idx,None])
                    
                if self.invCT2S==False: 
                    
                    mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                    mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                    mat3_array = np.concatenate([mat_m3[k] for k in mat_m3.keys()], axis=0)
                    
                    mat1_array = mat1_array[idx_unsort[0], :]
                    mat2_array = mat2_array[idx_unsort[1], :]
                    mat3_array = mat3_array[idx_unsort[2], :]
                    
                    self.synDat['T2S'] = mat1_array + mat2_array + mat3_array
                    
                    max_val = np.max(self.synDat['T2S'], axis=1, keepdims=True)
                    self.synDat['T2S'] = self.synDat['T2S']/max_val
                
                else:
                    synDat_m1 = np.matmul(m1,self.sysMatrix['T2S'].T)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace
                    synDat_m1 = np.multiply(synDat_m1, np.exp(-2j*np.pi*self.MW_f['T2S'][idx, None]*self.CT2S_TE/1000), dtype=np.complex128)
                    
                    synDat_m2 = np.matmul(m2,self.sysMatrix['T2S'].T)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace
                    synDat_m2 = np.multiply(synDat_m2, np.exp(-2j*np.pi*self.EW_f['T2S'][idx, None]*self.CT2S_TE/1000), dtype=np.complex128)
                    
                    synDat_m3 = np.matmul(m3,self.sysMatrix['T2S'].T)*(self.Inversion.T2Smax-self.Inversion.T2Smin)/self.Inversion.modSpace
                    synDat_m3 = np.multiply(synDat_m3, np.exp(-2j*np.pi*self.AW_f['T2S'][idx, None]*self.CT2S_TE/1000), dtype=np.complex128)

                    self.synDat['T2S'][idx]  = synDat_m1 + synDat_m2 + synDat_m3
                    self.synDat['T2S'][idx] *= np.exp(-1j*self.phi['T2S'][idx, None])     

                        
###############################################################################
# Überarbeitung notwendig: V0 and L1 could be tested on both single/joint inv !

    def fitness(self):
        
        '''
        Computes fitness values of all particles.
        --> quantifies model–observation agreement via objective function
        
        Parameters for use in PSO class initiation:
            invOpt [V0, V1] - method for joint inversion
                              V0: one single model vector for each signal\n
                              V1: one common model vector for both signals
            
            lpNorm [l1, L2] - Lp-Norm for generalization
                              L1: linear or L2: quadratic
                              
        Construction work: L1-Norm for single inversion can be tested, beta.
                           V0 not fully implemented yet!
        '''
        
        if self.lpNorm == 'L1':
            import sys
            sys.exit('ATTENTION: L1-Norm not yet implemeted.')
        
        if self.lpNorm == 'V0':
            import sys
            sys.exit('ATTENTION: Usage of multiple model vectors not yet implemeted.')
        
        
        if self.singleInv==True:
                
            sig = self.signType[0]
         
            if self.lpNorm == 'L1':
                self.fit[sig] = np.sum(np.abs(self.obsData[sig]-self.synDat[sig]), axis=1)/len(self.obsData[sig])
    
            if self.lpNorm == 'L2': # Frobenius norm:
                if self.invCT2S==False:
                    self.fit[sig] = np.sqrt(np.sum((self.obsData[sig]-self.synDat[sig])**2, axis=1))/len(self.obsData[sig])
                
                if self.invCT2S==True:
                    
                    self.fit[sig] = np.sqrt(np.sum((np.abs(self.obsData[sig])-np.abs(self.synDat[sig]))**2, axis=1))/len(self.obsData[sig]) + \
                                    np.sqrt(np.sum((np.angle(self.obsData[sig])-np.angle(self.synDat[sig]))**2, axis=1))/len(self.obsData[sig])
                                    
                    # sego's approach:
                    #   obsdata = np.concatenate(magnitude, phase) = sysnDat
                    
                    # obsDat_conc = np.concatenate([np.abs(self.obsData[sig]), np.angle(self.obsData[sig])])
                    # synDat_conc = np.concatenate([np.abs(self.synDat[sig]), np.angle(self.synDat[sig])], axis=1)
                    
                    # self.fit[sig] = np.sqrt(np.sum((obsDat_conc-synDat_conc)**2, axis=1))/len(obsDat_conc)
                
        if self.jointInv==True:
            
            weight_alpha, weight_beta, weight_gamma = 1, 1, 1
            
            term        = {sig:[] for sig in self.signType[0]}
            term['MWF'] = []
            keys        = term.keys()                      # e.g. T2, T2S & MWF
            
            for sig in self.signType[0]:

                term[sig] = np.sqrt(np.sum((self.obsData[sig]-self.synDat[sig])**2, axis=1))/len(self.obsData[sig])
                
                if sig == 'T2S' and self.invCT2S == True:
                    term['T2S'] = np.sqrt(np.sum((np.abs(self.obsData[sig])-np.abs(self.synDat[sig]))**2, axis=1))/len(self.obsData[sig]) + \
                                  np.sqrt(np.sum((np.angle(self.obsData[sig])-np.angle(self.synDat[sig]))**2, axis=1))/len(self.obsData[sig])
 
            term['MWF'] = (self.MWF[keys[0]]-self.MWF[keys[1]])**2
            
            self.fit[self.sigJoint] = weight_alpha*term[keys[0]] + weight_beta*term[keys[1]] + weight_gamma*term[keys[2]]    


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
        Updates the individual best positions (local bests) per particle.
        --> preserves personal best solutions across iterations
        
        Parameters for use in PSO class initiation:
            invOpt [V0, V1] - method for joint inversion
                              V0: one single model vector for each signal\n
                              V1: one common model vector for both signals
        
        Construction work: V1 should be tested and updated, beta.
        '''

        if self.singleInv == True:
            
            sig  = self.signType[0]
    
            fit  = self.fit[sig]
            bfit = self.bestFit[sig]
            mask = fit <= bfit
            
            self.bestFit[sig][mask]    = fit[mask]
            self.bestMWF[sig][mask]    = self.MWF[sig][mask]
            self.bestMod[sig][mask]    = self.mod[sig][mask]
            self.bestSynDat[sig][mask] = self.synDat[sig][mask]

        if self.jointInv == True:   

            i,j,k = self.signType[0], self.signType[1], self.sigJoint
            fit   = self.fit[k]
            bfit  = self.bestFit[k]
            mask  = fit <= bfit
            
            for sig in self.signType:

                self.bestMWF[sig][mask]    = self.MWF[sig][mask]
                self.bestMod[sig][mask]    = self.mod[sig][mask]
                self.bestSynDat[sig][mask] = self.synDat[sig][mask]
                
            self.bestFit[k][mask]          = self.fit[k][mask]
            self.bestMWF[k]                = np.sum([self.bestMWF[i],self.bestMWF[j]], axis=0)/len(self.signType)

###############################################################################

    def bestGlobal(self):

        '''
        Updates the global best position among all particles.
        --> drives convergence toward the swarm’s optimal solution
        '''

        if self.singleInv == True:
            
            sig       = self.signType[0]
            fit       = self.bestFit[sig]
            idx_min   = np.argmin(fit)
            val_min   = fit[idx_min]
        
            if val_min < self.globFit[sig]:
                self.globFit[sig]    = val_min
                self.globMod[sig]    = np.copy(self.mod[sig][idx_min])
                self.globSynDat[sig] = np.copy(self.bestSynDat[sig][idx_min])
                self.globIndex[sig]  = idx_min
        
        if self.jointInv == True:
            
            i,j,k     = self.signType[0], self.signType[1], self.sigJoint
            fit       = self.bestFit[k]
            idx_min   = np.argmin(fit)
            val_min   = fit[idx_min]

            if val_min < self.globFit[sig]:
                self.globFit[k]    = val_min
                self.globIndex[k]  = idx_min
                
                for sig in self.signType:
                    self.globSynDat[sig] = np.copy(self.bestSynDat[sig][idx_min])
                    self.globMod[sig]    = np.copy(self.mod[sig][idx_min])
        
###############################################################################

    def updatePos(self):
        
        '''
        Updates particle positions within the multidimensional search space
        --> position update governed by inertia, cognitive, and social components
        '''
        
        w  = self.PSO.w                               # inertia weight factor
        c1 = self.PSO.c1                              # social weight factor
        c2 = self.PSO.c2                              # cognitive weight factor

        for sig in self.signType:

            r = np.random.rand(self.noPart*2, self.noParam[sig])
            r11, r22 = r[::2], r[1::2]
            
            self.vel[sig] = w*self.vel[sig] + \
                            c1*r11*(self.bestMod[sig]-self.mod[sig]) + \
                            c2*r22*(self.globMod[sig]-self.mod[sig])
            
            self.mod[sig] = self.mod[sig] + self.vel[sig]
            
###############################################################################
    
    def checkLim(self):

        '''
        Enforces parameter boundaries for all particles
        --> uses np.clip() for efficient constraint handling
        '''
        
        for sig in self.signType:
            
            if self.noPeaks == 'GAUSS':

                self.m1[sig]     = np.clip(self.mod[sig][:,0], self.att.m1[0],     self.att.m1[1])
                self.m1_sig[sig] = np.clip(self.mod[sig][:,1], self.att.m1_sig[0], self.att.m1_sig[1])              
                self.m2[sig]     = np.clip(self.mod[sig][:,2], self.att.m2[0],     self.att.m2[1])
                self.m2_sig[sig] = np.clip(self.mod[sig][:,3], self.att.m2_sig[0], self.att.m2_sig[1])
                self.int2[sig]   = np.clip(self.mod[sig][:,4], self.att.int2[0],   self.att.int2[1])
                self.MWF[sig]    = np.clip(self.mod[sig][:,5], self.att.MWF[0],    self.att.MWF[1])
                
                if self.invCT2S==False:
                  
                    self.vel[sig] = np.where(np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig], 
                                                       self.m2_sig[sig], self.int2[sig],   self.MWF[sig]], 
                                                       axis=1) != self.mod[sig], 0, self.vel[sig])
            
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   
                                              self.m2_sig[sig], self.int2[sig],   self.MWF[sig]], axis=1)
                    
                if self.invCT2S==True:
                    
                    self.MW_f[sig] = np.clip(self.mod[sig][:,6], self.att.MW_f[0], self.att.MW_f[1])
                    self.FW_f[sig] = np.clip(self.mod[sig][:,7], self.att.FW_f[0], self.att.FW_f[1])
                    self.phi[sig]  = np.clip(self.mod[sig][:,8], self.att.phi[0],  self.att.phi[1])
                    
                    self.vel[sig] = np.where(np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig], 
                                                       self.m2_sig[sig], self.int2[sig],   self.MWF[sig],
                                                       self.MW_f[sig],   self.FW_f[sig],   self.phi[sig]],
                                                       axis=1) != self.mod[sig], 0, self.vel[sig])                    

                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   
                                              self.m2_sig[sig], self.int2[sig],   self.MWF[sig], 
                                              self.MW_f[sig],   self.FW_f[sig],   self.phi[sig]],  axis=1)
                    
        
            if self.noPeaks == 'DIRAC':
                
                self.m1[sig]     = np.clip(self.mod[sig][:,0], self.att.m1[0],      self.att.m1[1])
                self.m1_sig[sig] = np.clip(self.mod[sig][:,1], self.att.m1_sig[0],  self.att.m1_sig[1])              
                self.m2[sig]     = np.clip(self.mod[sig][:,2], self.att.m2[0],      self.att.m2[1])
                self.m2_sig[sig] = np.clip(self.mod[sig][:,3], self.att.m2_sig[0],  self.att.m2_sig[1])
                self.m3[sig]     = np.clip(self.mod[sig][:,4], self.att.m3[0],      self.att.m3[1])
                self.m3_sig[sig] = np.clip(self.mod[sig][:,5], self.att.m3_sig[0],  self.att.m3_sig[1])
                self.int2[sig]   = np.clip(self.mod[sig][:,6], self.att.int2[0],    self.att.int2[1])
                self.int3[sig]   = np.clip(self.mod[sig][:,7], self.att.int3[0],    self.att.int3[1])
                self.MWF[sig]    = np.clip(self.mod[sig][:,8], self.att.MWF[0],     self.att.MWF[1])
                
                if self.invCT2S==False:
                
                    self.vel[sig] = np.where(np.stack([self.m1[sig], self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig],
                                                       self.m3[sig], self.m3_sig[sig], self.int2[sig], self.int3[sig], 
                                                       self.MWF[sig]], axis=1) != self.mod[sig], 0, self.vel[sig])
                    
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig], self.m3[sig], 
                                              self.m3_sig[sig], self.int2[sig],   self.int3[sig], self.MWF[sig]],   axis=1)                              

                if self.invCT2S==True:
                
                    self.MW_f[sig] = np.clip(self.mod[sig][:,9],  self.att.MW_f[0], self.att.MW_f[1])
                    self.EW_f[sig] = np.clip(self.mod[sig][:,10], self.att.EW_f[0], self.att.EW_f[1])
                    self.AW_f[sig] = np.clip(self.mod[sig][:,11], self.att.AW_f[0], self.att.AW_f[1])                   
                    self.phi[sig]  = np.clip(self.mod[sig][:,12], self.att.phi[0],  self.att.phi[1])
                        
                    self.vel[sig] = np.where(np.stack([self.m1[sig],   self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig],
                                                       self.m3[sig],   self.m3_sig[sig], self.int2[sig], self.int3[sig], 
                                                       self.MWF[sig],  self.MW_f[sig],   self.EW_f[sig], self.AW_f[sig], 
                                                       self.phi[sig]], axis=1) != self.mod[sig], 0, self.vel[sig])
                    
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig], self.m3[sig], 
                                              self.m3_sig[sig], self.int2[sig],   self.int3[sig], self.MWF[sig],    self.MW_f[sig], 
                                              self.EW_f[sig],   self.AW_f[sig],   self.phi[sig]],  axis=1)
                    

###############################################################################

    def init_sysMatrix(self, rootMWF, **kwargs):
        
        yy, xx = kwargs.get('position', (0,0))
        
        
        self.sysMatrix['T1']  = rootMWF['T1_MATRIX']  if self.invT1==True else []
        self.sysMatrix['T2']  = rootMWF['T2_MATRIX']  if self.invT2==True else []
        self.sysMatrix['T2S'] = rootMWF['T2S_MATRIX'] if self.invT2S==True else []
        
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
        
    def init_PS(self):
        
        # Generate synthetic data based on the current particle model parameters
        self.createSynData_fast(ind_iter=999)
        
        # Compute fitness values of all particles
        self.fitness()

        # Update the individual best positions (local bests) per particle
        self.bestLocal()

        # Update the global best position among all particles
        self.bestGlobal()
        
        # # collect best global fit and indicies for a number of n iterations
        # if self.plotResult==True and self.singleInv==True:
            
        #     sig                    = self.signType[0]
        #     self.globInd_list      = {'T1': [], 'T2': [], 'T2S': [], 'CT2S': []}
        #     self.globFit_list      = {'T1': [], 'T2': [], 'T2S': [], 'CT2S': []}
        #     self.globMWF_list      = {'T1': [], 'T2': [], 'T2S': [], 'CT2S': []}
        #     self.globInd_list[sig] = [self.globIndex[sig]+1]
        #     self.globFit_list[sig] = [self.globFit[sig]]
        #     self.globMWF_list[sig] = [self.globMod[sig][-1]]
        
###############################################################################  
    
    def execPSO(self, obsData: np.array, plotIterTest=False, callback_plot=None):

        self.plotResult = plotIterTest
        self.obsData    = obsData
        
        tt = time.time()
        self.init_PS()
        self.log(tt, 'ms')
        
        if self.config.PSO_spec.dyn.pixel == [45,55]:            
            times = np.zeros((self.noIter, 6))
        
        for jj in range(self.noIter)[:]:
            
            if self.config.PSO_spec.dyn.pixel == [45,55]:
                # print(f'\nITERATION {jj} \n')
                
                start = time.perf_counter()
                self.updatePos()
                times[jj, 0] = (time.perf_counter() - start) * 1e3
                # print(f"updatePos: {(time.perf_counter() - start)*1e6:.2f} mus")
                
                start = time.perf_counter()
                self.checkLim()
                times[jj, 1] = (time.perf_counter() - start) * 1e3
                # print(f"checkLim: {(time.perf_counter() - start)*1e6:.2f} mus")
                
                start = time.perf_counter()
                self.createSynData_fast(ind_iter=jj)
                times[jj, 2] = (time.perf_counter() - start) * 1e3
                # print(f"createSynData_fast: {(time.perf_counter() - start)*1e6:.2f} mus")
                
                start = time.perf_counter()
                self.fitness()
                times[jj, 3] = (time.perf_counter() - start) * 1e3
                # print(f"fitness: {(time.perf_counter() - start)*1e6:.2f} mus")
                
                start = time.perf_counter()
                self.bestLocal()
                times[jj, 4] = (time.perf_counter() - start) * 1e3
                # print(f"bestLocal: {(time.perf_counter() - start)*1e6:.2f} mus")
                
                start = time.perf_counter()
                self.bestGlobal()
                times[jj, 5] = (time.perf_counter() - start) * 1e3
                # print(f"bestGlobal: {(time.perf_counter() - start)*1e6:.2f} mus")

            else:
                # update particle position in the global search space
                # each Particle  -->  locVector = w*v1 + c1*r1*(m[i+1]-m[i])              
                self.updatePos()
                
                # # check the limits of new model values
                # # --> using better np.clip() funtion instead of generic expressions            
                self.checkLim()
        
                # update synthetic data set
                self.createSynData_fast(ind_iter=jj)
                            
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

                if callback_plot is not None and jj in [1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99]:
                    string = [self.signType[0], str(jj).zfill(3)]
                    callback_plot(PSO=self, string=string)                    
                    
        # if self.config.PSO_spec.dyn.pixel == [45,55]:  
        #     total_times = np.sum(times, axis=0)
        #     functions = ["updatePos", "checkLim", "createSynData_fast", "fitness", "bestLocal", "bestGlobal"]
            
        #     for i,j in enumerate(total_times):
        #         print(f'{functions[i]}: {int(j)} ms')


###############################################################################
###############################################################################
###############################################################################
###############################################################################

    def plot_model(self, model_array: np.array, no_curves: int, ind_iter: str):
        
        '''
        Plots a random number of gaussian models.
        '''
        
        # random_indices = np.random.choice(model_array.shape[0], size=no_curves, replace=False)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(no_curves): #random_indices:
            y = model_array[i]
            x = np.arange(len(y))
        
            # Labels für zusammenhängende Bereiche mit y != 0
            labels, n_peaks = label(y >= 0.001)
        
            # Plot alle Null-Werte als graue Linie
            idx_zero = y == 0
            ax.plot(x[idx_zero], y[idx_zero], color='grey', alpha=0.5)
        
            # Farben für die Peaks definieren
            colors = ['blue', 'red', 'green']
        
            # Plot jeden Peak separat
            for peak_id in range(1, n_peaks + 1):
                idx = labels == peak_id
                if peak_id <= len(colors):  # Nur 3 Farben zuweisen
                    ax.plot(x[idx], y[idx], color=colors[peak_id - 1])
        
            # Plot alle Null-Werte als graue Linie
            idx_zero = y <= 0
            ax.plot(x[idx_zero], y[idx_zero], color='grey', alpha=0.5, linewidth=0.25)
            
        # Titel oben mittig   
        pix  = self.config.PSO_spec.dyn.pixel
        _t1_ = 'MW (blue), EW (red), AW (green)' if self.noPeaks == 'DIRAC' else 'MW (blue), FW (red)'
       
        if ind_iter == 999:
            _t2_ = 'Start'
            _sp_ = f'{self.save_path}model_start_x{pix[1]}y{pix[0]}.png'
        else:
            _t2_ = f'Iter {ind_iter+1}'
            _sp_ = f'{self.save_path}model_iter{str(ind_iter).zfill(3)}_x{pix[1]}y{pix[0]}.png'
            
        # if ind_iter == 0:
        #     _t2_ = 'Iter 001'
        #     _sp_ = f'{self.save_path}model_iter001_x{pix[1]}y{pix[0]}.png'
        # elif ind_iter == 999:
        #     _t2_ = 'Start'
        #     _sp_ = f'{self.save_path}model_start_x{pix[1]}y{pix[0]}.png'
        # else:
        #     _t2_ = f'Iter {ind_iter+1}'
        #     _sp_ = f'{self.save_path}model_final_x{pix[1]}y{pix[0]}.png'
    
        text = f'{_t2_}; pix.{pix}: random {no_curves} gaussian model curves for {_t1_}'
        
        ax.set_xlim(-50, 1050)
        ax.text(0.5, 0.9, text, transform=ax.transAxes, ha='center', 
                va='bottom', fontsize=11, fontweight='bold')
        
        os.makedirs(self.save_path, exist_ok=True)
        
        fig.savefig(_sp_, dpi=300, bbox_inches='tight')
        
        plt.close()

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
                                "_PSO version": self.config.general.PSO_vs,
                                "_Signal type": self.signType,
                                "_Iterations":  self.noIter,
                                "_Particles":   self.noPart,                            
                                "_PSO cycles":  self.noPSOIter,
                                "_w":           self.PSO.w,
                                "_c1":          self.PSO.c1,
                                "_c2":          self.PSO.c2,
                                "_Lp Norm":     self.lpNorm},
                            # "Inversion specifications":{
                            #     "_T2_min":      self.Inversion.T2min,
                            #     "_T2max":       self.Inversion.T2max}
                            "ObsData":{
                                "_Source":               self.dataSource,
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
                                "_MWF":    getattr(getattr(self, self.signType[0]), self.noPeaks).MWF,
                                "_MW_f":   None,
                                "_FW_f":   None,
                                "_EW_f":   None,
                                "_AW_f":   None,
                                "_phi":    None
                            }}
            
            if self.noPeaks == 'DIRAC':
                jsonData[f"SynData{self.signType[0]}"]["_m3"]     = getattr(getattr(self, self.signType[0]), self.noPeaks).m3
                jsonData[f"SynData{self.signType[0]}"]["_m3_sig"] = getattr(getattr(self, self.signType[0]), self.noPeaks).m3_sig
                jsonData[f"SynData{self.signType[0]}"]["_integ3"] = getattr(getattr(self, self.signType[0]), self.noPeaks).int3
                
                if self.invCT2S == True:
                    jsonData[f"SynData{self.signType[0]}"]["_MW_f"] = getattr(getattr(self, self.signType[0]), self.noPeaks).MW_f
                    jsonData[f"SynData{self.signType[0]}"]["_EW_f"] = getattr(getattr(self, self.signType[0]), self.noPeaks).EW_f
                    jsonData[f"SynData{self.signType[0]}"]["_AW_f"] = getattr(getattr(self, self.signType[0]), self.noPeaks).AW_f
                    jsonData[f"SynData{self.signType[0]}"]["_phi"]  = getattr(getattr(self, self.signType[0]), self.noPeaks).phi
                    
            if self.noPeaks == 'GAUSS' and self.invCT2S == True:
                
                jsonData[f"SynData{self.signType[0]}"]["_MW_f"] = getattr(getattr(self, self.signType[0]), self.noPeaks).MW_f
                jsonData[f"SynData{self.signType[0]}"]["_FW_f"] = getattr(getattr(self, self.signType[0]), self.noPeaks).FW_f
                jsonData[f"SynData{self.signType[0]}"]["_phi"]  = getattr(getattr(self, self.signType[0]), self.noPeaks).phi
                
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
                                "_Source":               self.dataSource,
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
    
    def result2array(self, results:dict, result_map: np.array, syn_dat_map: np.array, res_array_dic: np.array, kk:int, **kwargs):
        
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
                
                if self.invCT2S == True:
                    syn_dat_map[sig] = syn_dat_map[sig].astype(np.complex128)
                    
        #######################################################################
                # synthetic data save
                for item in results:
                    y, x = item['pix']
                    
                    syn_dat_map[sig][y, x, :, kk] = item['synDat']
                    res_array_dic[y, x, kk]       = item
        ####################################################################### 
                
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
            
        return result_map, syn_dat_map, res_array_dic
        
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
            
            if t_ms >0:
                print(f'{string}: {t_ms} ms')
        
        if dim=='mus':
            t_mus        = round(T_elapsed*1000 * 1000, 2)
            print(f'{string}: {t_mus} mus')#'\u03BCs')
            # if t_mus > 0:
            #     print(f'{string}: {t_mus} mus')#'\u03BCs')