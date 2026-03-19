# SPDX-FileCopyrightText: 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Authors:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#   Ségolène Dega (Helmholtz Centre for Environmental Research - UFZ)
#   Hendrik Paasche (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of PyMyelinPSO.
# See the LICENSE file in the project root for license information.

"""
Class for applying particle swarm optimization (PSO) to in vivo MRI data,
including PSO methods and additional PSO-related helper functions.
"""

import time, os, copy, json
import numpy                as     np
from   numba                import njit

import help_tools           as     hlp
from   pso_model_parameters import Parameters   as PM


###############################################################################
# Numba-accelerated core functions for PSO-based MRI signal simulation and
# non-numba equivalents. For max speed use/try: @njit(fastmath=True, cache=True)

@njit
def _compute_gaussian_njit(x, mean, fac_sig_sqrt, fac_pre):    
    """Internal Gaussian basis kernel evaluated on the system grid."""    
    dx  = x - mean
    return fac_pre * np.exp(-np.square(dx) * fac_sig_sqrt)

def _compute_gaussian_py(x, mean, fac_sig_sqrt, fac_pre):    
    """Internal Gaussian basis kernel evaluated on the system grid."""    
    dx  = x - mean
    return fac_pre * np.exp(-np.square(dx) * fac_sig_sqrt) 

@njit
def _compute_matmul_njit(gauss, sys_mat):    
    """Internal matrix multiplication kernel."""    
    return gauss @ sys_mat

def _compute_matmul_py(gauss, sys_mat):    
    """Internal matrix multiplication kernel."""    
    return gauss @ sys_mat

@njit
def _compute_CT2S_njit(freq, phi, m_arr, mult_CT2S, n_comp=2):
    """Internal kernel for complex CT2* signal computation."""                   
    exp_MW  = np.exp(mult_CT2S * freq[0])        # frequency shift MW component
    exp_phi = np.exp(-1j * phi)                  # global phase shift    
    if n_comp == 2:
        exp_FW = np.exp(mult_CT2S * freq[1])     # frequency shift FW component
        return (m_arr[0]*exp_MW + m_arr[1]*exp_FW) * exp_phi    
    elif n_comp == 3:
        exp_EW = np.exp(mult_CT2S * freq[1])     # frequency shift EW component
        exp_AW = np.exp(mult_CT2S * freq[2])     # frequency shift AW component
    return (m_arr[0]*exp_MW + m_arr[1]*exp_EW + m_arr[2]*exp_AW) * exp_phi

def _compute_CT2S_py(freq, phi, m_arr, mult_CT2S, n_comp=2):
    """Internal kernel for complex CT2* signal computation."""                   
    exp_MW  = np.exp(mult_CT2S * freq[0])        # frequency shift MW component
    exp_phi = np.exp(-1j * phi)                  # global phase shift    
    if n_comp == 2:
        exp_FW = np.exp(mult_CT2S * freq[1])     # frequency shift FW component
        return (m_arr[0]*exp_MW + m_arr[1]*exp_FW) * exp_phi    
    elif n_comp == 3:
        exp_EW = np.exp(mult_CT2S * freq[1])     # frequency shift EW component
        exp_AW = np.exp(mult_CT2S * freq[2])     # frequency shift AW component
    return (m_arr[0]*exp_MW + m_arr[1]*exp_EW + m_arr[2]*exp_AW) * exp_phi


###############################################################################

class ParticleSwarmOptimizer(PM):
        
    '''
    Particle swarm optimizer for fitting in vivo MRI relaxation data (T2, T2*).
    Optionally extendable to also invert T1 signal - please contact the author.
    
    Args:
        config_data (dict): configuration parameters
    '''
        
    def __init__(self, config_data: dict, **kwargs):

        # get optional kwargs if model
        self.init_matrix = kwargs.get('init_matrix', False)
        self.sys_param   = kwargs.get('sys_param', dict())
        self.position    = kwargs.get('position', (0,0))
        self.constants   = kwargs.get('constants', dict())
        
        # config to class object
        self._config_to_object(config_data=config_data)
        
        # Inherit PSO parameters for T1, T2, and T2* signals from PM class
        super().__init__()

        # Definitions related to algo_math
        # --> computational settings for optimal algorithm performance
        self.use_njit        = self.config.PSO_spec.algo_math.njit
        self.width_gauss     = self.config.PSO_spec.algo_math.width
        self.batch_size      = self.config.PSO_spec.algo_math.batch

        # Definitions related to PSO_math
        # --> PSO-specific mathematical adjustments
        self.lp_norm         = self.config.PSO_spec.PSO_math.lp_norm
        self.weights         = self.config.PSO_spec.PSO_math.weights
        self.n_mod_vec       = self.config.PSO_spec.PSO_math.n_mod_vec
        self.n_comp          = self.config.PSO_spec.PSO_math.n_comp
        self.n_iter          = self.config.PSO_spec.PSO_math.n_iter
        self.n_part          = self.config.PSO_spec.PSO_math.n_part
        self.norm_max        = self.config.PSO_spec.PSO_math.norm
        self.rand_seed_bool  = self.config.PSO_spec.PSO_math.rand[0]
        self.rand_seed_value = self.config.PSO_spec.PSO_math.rand[1]
        
        if self.rand_seed_bool:
            np.random.seed(self.rand_seed_value)
        
        if self.config.PSO_spec.comp_mode.PSO_on_slice.use:
            self.n_pso_cycles = self.config.PSO_spec.PSO_math.cyc_slice
        else:
            self.n_pso_cycles = self.config.PSO_spec.PSO_math.cyc_pixel
        
        # Source signal selection and inversion choice
        self.inv_T1      = self.config.source.signal.T1
        self.inv_T2      = self.config.source.signal.T2
        self.inv_T2S     = self.config.source.signal.T2S
        self.inv_CT2S    = self.config.source.signal.CT2S
        self.inv_SI      = self.config.source.signal.SI
        self.inv_JI      = self.config.source.signal.JI
        self.data_source = self.config.source.data.type
        self.add_noise   = self.config.source.data.add_noise[0]
        self.SNR         = self.config.source.data.add_noise[1]   
        self.decay_types = [sig for sig, inv in zip(['T1','T2','T2S'], 
                           [self.inv_T1, self.inv_T2, self.inv_T2S]) if inv]
                   
        if self.inv_JI:
            self.decay_JI = '_'.join([self.decay_types[0], 'CT2S'] if self.inv_CT2S else self.decay_types)
        else:
            self.decay_JI = 'None'
        
        # Brain regions for analysis
        brain_ana      = ['CSF', 'GM', 'WM', 'dGM', 'BS', 'CB']
        self.brain_regions = "_".join([brain_ana[i] for i in [x - 1 for x in self.config.source.mask.seg]])
        
        # Dictionaries for inversion/integration-parameters for T1, T2 and T2**     
        # MxN means in the context of numpy M rows and N columns
        self.att        = {'T1':[],'T2':[],'T2S':[]}
        self.sys_matrix = {'T1':[],'T2':[],'T2S':[]}
        self.sys_grid   = {'T1':[],'T2':[],'T2S':[]}
        self.m1         = {'T1':[],'T2':[],'T2S':[]}
        self.m1_sig     = {'T1':[],'T2':[],'T2S':[]}
        self.m2         = {'T1':[],'T2':[],'T2S':[]}
        self.m2_sig     = {'T1':[],'T2':[],'T2S':[]}
        self.m3         = {'T1':[],'T2':[],'T2S':[]}
        self.m3_sig     = {'T1':[],'T2':[],'T2S':[]}
        self.int2       = {'T1':[],'T2':[],'T2S':[]}
        self.int3       = {'T1':[],'T2':[],'T2S':[]}
        self.MWF        = {'T1':[],'T2':[],'T2S':[],self.decay_JI:[]}
        self.MW_f       = {'T2S': []} # myelin water       - frequency shift
        self.FW_f       = {'T2S': []} # free water         - frequency shift
        self.AW_f       = {'T2S': []} # axonal water       - frequency shift
        self.EW_f       = {'T2S': []} # extracelular water - frequency shift
        self.phi        = {'T2S': []} # global phase shift
        
        # Dictionaries for PSO parameters
        self.n_param    = {'T1':[],'T2':[],'T2S':[]}
        self.mod_help   = {'T1':[],'T2':[],'T2S':[]}
        self.vel        = {'T1':[],'T2':[],'T2S':[]}
        self.fit        = {'T1':[],'T2':[],'T2S':[],self.decay_JI:[]}
        self.syn_decay  = {'T1':[],'T2':[],'T2S':[]}
        self.n_echoes   = {'T1':   self.Inv.n_echoes_T1,
                           'T2':   self.Inv.n_echoes_T2,
                           'T2S':  self.Inv.n_echoes_T2S}

        # Filling dictionaries with content
        for signal in self.decay_types:
            
            self._build_model_vector(signal)     
            self.mod             = copy.deepcopy(self.mod_help)
            
            self.vel[signal]     = np.random.uniform(-0.3, 0.3, (self.n_part, self.n_param[signal]))*self.mod[signal]
            
            if self.inv_SI:
                self.fit[signal] = np.empty(self.n_part)
                
            if self.inv_CT2S:
                self.syn_decay[signal] = np.empty((self.n_part, self.n_echoes[signal]), dtype=np.complex128)
            else:  # T1, T2, T2S
                self.syn_decay[signal] = np.empty((self.n_part, self.n_echoes[signal]), dtype=np.float64)
        
        # Continue Filling dictionaries with content if joint inversion was chosen
        if self.inv_JI:
            self.fit[self.decay_JI] = np.empty(self.n_part)
            self.MWF[self.decay_JI] = np.empty(self.n_part)
            
        # Dictionaries for a best local PSO results       
        self.best_MWF      = {key: np.full_like(self.MWF[key], np.inf) for key in self.MWF.keys()}
        self.best_fit      = {key: np.full_like(self.fit[key], np.inf) for key in self.fit.keys()}
        self.best_mod      = {key: np.full_like(self.mod[key], np.inf) for key in self.mod.keys()}
        self.best_syn_data = {key: np.full_like(self.syn_decay[key], np.inf) for key in self.syn_decay.keys()}
        
        # Dictionaries for a best global PSO results           
        self.glob_fit      = {'T1':np.inf,'T2':np.inf,'T2S':np.inf,self.decay_JI:np.inf}
        self.glob_mod      = {'T1':[],'T2':[],'T2S':[]}
        self.glob_syn_data = {'T1':[],'T2':[],'T2S':[]}
        self.glob_ind      = {'T1':[],'T2':[],'T2S':[],self.decay_JI:[]}
        
        # Calculation of constants used in the PSO methods
        if self.constants and self.init_matrix:
          
            # forward solver (fs) constant factors
            self.fs_divisor_T1  = self.constants['fs_divisor_T1']
            self.fs_divisor_T2  = self.constants['fs_divisor_T2']
            self.fs_divisor_T2S = self.constants['fs_divisor_T2S']
            
            self.init_system_matrix(root_MWF=self.sys_param, position=self.position)

            if self.inv_T1:
                self.fs_sysMatr_T1  = np.ascontiguousarray(self.sys_matrix['T1'].T*self.fs_divisor_T1)
            if self.inv_T2:
                self.fs_sysMatr_T2  = np.ascontiguousarray(self.sys_matrix['T2'].T*self.fs_divisor_T2)
            if self.inv_T2S:
                self.fs_sysMatr_T2S = np.ascontiguousarray(self.sys_matrix['T2S'].T*self.fs_divisor_T2S)
            if self.inv_CT2S:
                self.mult_CT2S   = self.constants['mult_CT2S']
                self.CT2S_TE     = self.sys_param['CT2S_TE']
            
            self.size_batches    = self.constants['size_batches']
            self.num_batches     = self.constants['num_batches']
            self.idx_slices      = self.constants['idx_slices']
            self.idx_complex     = self.constants['idx_complex']
            self.factor_gauss    = self.constants['factor_gauss']
            self.sys_grid['T1']  = self.constants['sys_grid_T1']
            self.sys_grid['T2']  = self.constants['sys_grid_T2']
            self.sys_grid['T2S'] = self.constants['sys_grid_T2S']
        
        # Bind in numba-accelerated core functions
        self._compute_gaussian = _compute_gaussian_njit if self.use_njit else _compute_gaussian_py
        self._compute_matmul   = _compute_matmul_njit   if self.use_njit else _compute_matmul_py
        self._compute_CT2S_sig = _compute_CT2S_njit     if self.use_njit else _compute_CT2S_py

###############################################################################

    def __del__(self):
        """Explicitly releases internal references."""
        self._close()
    
    def _close(self):
        self.__dict__.clear()

###############################################################################

    def _config_to_object(self, config_data: dict):
        
        """Transfers configuration dictionary to attribute-based object structure."""
        
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

    def _config_to_attribute(self, signal):
        
        """Stores initial model parameters for a given MRI signal."""
        
        if self.n_comp == 2:
            self.param_class      = "TwoComponentParams"
        if self.n_comp == 3:
            self.param_class      = "ThreeComponentParams"
            
        obj              = getattr(self, signal)
        self.att[signal] = getattr(obj, self.param_class)        

###############################################################################
    
    def _constant_PSO_objects(self, CT2S_TE):
        
        """Computes and returns constant PSO-related parameters."""
   
        const = {
                "fs_divisor_T1":  (self.Inv.T1_max - self.Inv.T1_min) / self.Inv.mod_space,
                "fs_divisor_T2":  (self.Inv.T2_max - self.Inv.T2_min) / self.Inv.mod_space,
                "fs_divisor_T2S": (self.Inv.T2S_max - self.Inv.T2S_min) / self.Inv.mod_space,
                "size_batches":   self.config.PSO_spec.algo_math.batch,
                "num_batches":    self.n_part // self.config.PSO_spec.algo_math.batch,
                "idx_slices":     [slice(i*self.config.PSO_spec.algo_math.batch, (i+1)*self.config.PSO_spec.algo_math.batch) 
                                   for i in range(self.n_part // self.config.PSO_spec.algo_math.batch)],
                "idx_complex":    [slice(i*self.config.PSO_spec.algo_math.batch*2, (i+1)*self.config.PSO_spec.algo_math.batch*2) 
                                   for i in range(self.n_part // (self.config.PSO_spec.algo_math.batch*2))],
                "width_gauss":    self.config.PSO_spec.algo_math.width,
                "factor_gauss":   np.sqrt(2*np.pi),
                "sys_grid_T1":    np.linspace(self.Inv.T1_min, self.Inv.T1_max, self.Inv.mod_space),
                "sys_grid_T2":    np.linspace(self.Inv.T2_min, self.Inv.T2_max, self.Inv.mod_space),
                "sys_grid_T2S":   np.linspace(self.Inv.T2S_min, self.Inv.T2S_max, self.Inv.mod_space),
                }

        if self.inv_CT2S:
            const["mult_CT2S"] = -2j * np.pi * CT2S_TE / 1000
    
        return const
        
###############################################################################
      
    def _build_model_vector(self, signal):
        
        """Initializes particle model vectors based on PSO parameter ranges."""

        # # modelvector-matrix in the shape 3xMxN --> 3 signals, M particles, N parameters
        # #    mod = [[[mod11][mod12][.....][mod1N]]
        # #           [[mod21][mod12][.....][mod1N]]
        # #           [[.....][.....][.....][.....]]
        # #           [[modM1][modM2][.....][modMN]]]
        
        self._config_to_attribute(signal)

        self.m1[signal]     = np.random.uniform(self.att[signal].m1[0],     self.att[signal].m1[1],     self.n_part)
        self.m1_sig[signal] = np.random.uniform(self.att[signal].m1_sig[0], self.att[signal].m1_sig[1], self.n_part)
        self.m2[signal]     = np.random.uniform(self.att[signal].m2[0],     self.att[signal].m2[1],     self.n_part)
        self.m2_sig[signal] = np.random.uniform(self.att[signal].m2_sig[0], self.att[signal].m2_sig[1], self.n_part)
        self.int2[signal]   = np.random.uniform(self.att[signal].int2[0],   self.att[signal].int2[1],   self.n_part)
        self.MWF[signal]    = np.random.uniform(self.att[signal].MWF[0],    self.att[signal].MWF[1],    self.n_part)

        self.mod_help[signal]  = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                  self.m2_sig[signal], self.int2[signal],   self.MWF[signal]))
                
        if self.inv_CT2S and signal == 'T2S' and self.n_comp == 2:
            
            self.MW_f[signal]  = np.random.uniform(self.att[signal].MW_f[0], self.att[signal].MW_f[1], self.n_part)
            self.FW_f[signal]  = np.random.uniform(self.att[signal].FW_f[0], self.att[signal].FW_f[1], self.n_part)
            self.phi[signal]   = np.random.uniform(self.att[signal].phi[0],  self.att[signal].phi[1],  self.n_part)
            
            self.mod_help[signal] = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                     self.m2_sig[signal], self.int2[signal],   self.MWF[signal],
                                                     self.MW_f[signal],   self.FW_f[signal],   self.phi[signal]))
        
        if self.n_comp == 3:

            self.m3[signal]     = np.random.uniform(self.att[signal].m3[0],     self.att[signal].m3[1],     self.n_part)
            self.m3_sig[signal] = np.random.uniform(self.att[signal].m3_sig[0], self.att[signal].m3_sig[1], self.n_part)
            self.int3[signal]   = np.random.uniform(self.att[signal].int3[0],   self.att[signal].int3[1],   self.n_part)

            self.mod_help[signal]  = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                      self.m2_sig[signal], self.m3[signal],     self.m3_sig[signal],
                                                      self.int2[signal],   self.int3[signal],   self.MWF[signal]))
        
        if self.inv_CT2S and signal == 'T2S' and self.n_comp == 3:
            
            self.MW_f[signal]  = np.random.uniform(self.att[signal].MW_f[0], self.att[signal].MW_f[1], self.n_part)
            self.EW_f[signal]  = np.random.uniform(self.att[signal].EW_f[0], self.att[signal].EW_f[1], self.n_part)
            self.AW_f[signal]  = np.random.uniform(self.att[signal].AW_f[0], self.att[signal].AW_f[1], self.n_part)
            self.phi[signal]   = np.random.uniform(self.att[signal].phi[0],  self.att[signal].phi[1],  self.n_part)
            
            self.mod_help[signal] = np.column_stack((self.m1[signal],     self.m1_sig[signal], self.m2[signal],   
                                                     self.m2_sig[signal], self.m3[signal],     self.m3_sig[signal],
                                                     self.int2[signal],   self.int3[signal],   self.MWF[signal],  
                                                     self.MW_f[signal],   self.EW_f[signal],   self.AW_f[signal], self.phi[signal]))

        self.n_param[signal] = self.mod_help[signal].shape[-1]

###############################################################################

    # Gaussian basis functions used for synthetic decay curves
    #
    # a) Full integration over the entire system grid without any acceleration
    #
    # --> Performance strongly dependent on elementwise np.exp and np.sqrt
    
    def _compute_gauss_full(self, x, sigma, mean, scale):        
        """Computes fully integrated Gaussian bell curves on the system grid."""
        return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
    
    # b) Preparation for accelerated integration using a region of interest (ROI)
    
    # --> Gaussian integration is restricted to index intervals where
    #     contributions exceed a fixed threshold of 1e-3 (see README.md)
    
    def _compute_constants_gauss(self, sigma, scale):      
        
        """Pre-computes constant factors for Gaussian integration.""" 
             
        _sigma_square_inv = 1 / (2 * sigma ** 2)
        _prefactor        = scale / (self.factor_gauss * sigma)        
        return _sigma_square_inv, _prefactor
 
    def _compute_intervall_gauss(self, sigma, mean, scale, signal='T2'):
        
        """
        Computes batch-wise index intervals on the system grid used for
        Gaussian integration restricted to intervals of interest.
        """
        
        max_sig   = sigma.reshape(self.num_batches, self.size_batches).max(axis=1)
        max_scale = scale.reshape(self.num_batches, self.size_batches).max(axis=1)
      
        with np.errstate(divide='ignore', invalid='ignore'):
            inverse1 = 1e-3 * np.sqrt(2 * np.pi) * max_sig / max_scale
            inverse2 = np.sqrt(-2 * max_sig**2 * np.log(inverse1))
            inverse2 = np.where(np.isfinite(inverse2), inverse2, 0)
        
        _left     = mean.reshape(self.num_batches, self.size_batches).min(axis=1) - inverse2
        _right    = mean.reshape(self.num_batches, self.size_batches).max(axis=1) + inverse2        
        left      = np.searchsorted(self.sys_grid[signal], _left, side='left')   - 1
        right     = np.searchsorted(self.sys_grid[signal], _right, side='right') + 1
        
        left[left < 0] = 0
    
        return left, right  


    def compute_synthetic_decay(self):

        """ 
        Computes synthetic MRI decay signals (T2, T2S, CT2S) for a number of particles.            
        The method is matrix-based on integration over two or three Gaussians.
        
        Returns:
            Dictionary .synDat with np.ndarrays in the shape [noPart, signal length]. 
        
        """

        # NOTE 1: compared to the functions to compute Gaussians, arrays are:
        #         values ==> mean ; weights ==> scale; widths ==> sigma
        
        # NOTE 2: the whole function can be condensed if wished, which is not
        #         be done yet for reasons of readability and traceability
        
        if self.n_comp == 2:

            if self.inv_T1:
                
                values    = np.array([self.m1['T1'],self.m2['T1']])
                weights   = np.array([self.int2['T1']*self.MWF['T1']/(1-self.MWF['T1']),self.int2['T1']])
                widths    = np.array([self.m1_sig['T1'],self.m2_sig['T1']])      
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2 = {}, {}
                
                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights) 
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T1')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T1')
                    
                for i, idx in enumerate(self.idx_slices):

                        m1 = self._compute_gaussian(self.sys_grid['T1'][None, left_m1[i]:right_m1[i]], 
                                                    values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = self._compute_gaussian(self.sys_grid['T1'][None, left_m2[i]:right_m2[i]], 
                                                    values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])

                        mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T1[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T1[left_m2[i]:right_m2[i], :])
                        
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                
                self.syn_decay['T1'] = mat1_array + mat2_array
                
                if self.norm_max:
                    max_val = np.max(self.syn_decay['T1'], axis=1, keepdims=True)
                    self.syn_decay['T1'] = self.syn_decay['T1']/max_val      
                        
            if self.inv_T2:
                
                values    = np.array([self.m1['T2'],self.m2['T2']])
                weights   = np.array([self.int2['T2']*self.MWF['T2']/(1-self.MWF['T2']),self.int2['T2']])
                widths    = np.array([self.m1_sig['T2'],self.m2_sig['T2']])      
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2 = {}, {}
                
                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights) 
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2')
                    
                for i, idx in enumerate(self.idx_slices):

                        m1 = self._compute_gaussian(self.sys_grid['T2'][None, left_m1[i]:right_m1[i]], 
                                                    values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                        m2 = self._compute_gaussian(self.sys_grid['T2'][None, left_m2[i]:right_m2[i]], 
                                                    values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])

                        mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :])
                        mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :])
                        
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                
                self.syn_decay['T2'] = mat1_array + mat2_array
                
                if self.norm_max:
                    max_val = np.max(self.syn_decay['T2'], axis=1, keepdims=True)
                    self.syn_decay['T2'] = self.syn_decay['T2']/max_val                

            if self.inv_T2S:

                values    = np.array([self.m1['T2S'],self.m2['T2S']])
                weights   = np.array([self.int2['T2S']*self.MWF['T2S']/(1-self.MWF['T2S']),self.int2['T2S']])
                widths    = np.array([self.m1_sig['T2S'],self.m2_sig['T2S']])        
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2 = {}, {}

                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2S')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2S')
                                
                for i, idx in enumerate(self.idx_slices):
                        
                    m1 = self._compute_gaussian(self.sys_grid['T2S'][None, left_m1[i]:right_m1[i]], 
                                                values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                    m2 = self._compute_gaussian(self.sys_grid['T2S'][None, left_m2[i]:right_m2[i]], 
                                                values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                    
                    mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :])
                    mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :])

                ## version 2.0.0 alpha
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                    
                if not self.inv_CT2S:
                    self.syn_decay['T2S'] = mat1_array + mat2_array
                    
                    if self.norm_max:
                        max_val = np.max(self.syn_decay['T2S'], axis=1, keepdims=True)
                        self.syn_decay['T2S'] = self.syn_decay['T2S']/max_val
                
                else:
                    # # NOTE: this function is the actual bottleneck!
                    
                    # # shape syn_array -> (n_part, n_echoes)
                    # for i, idx in enumerate(self.idx_complex):
                    #     self.syn_decay['T2S'][idx] = self._compute_CT2S_sig(freq      = [self.MW_f['T2S'][idx, None], 
                    #                                                                      self.FW_f['T2S'][idx, None]], 
                    #                                                         phi       =  self.phi['T2S'][idx, None], 
                    #                                                         m_arr     = [mat1_array[idx], mat2_array[idx]], 
                    #                                                         mult_CT2S = self.mult_CT2S,
                    #                                                         n_comp    = self.n_comp)
                    
                    self.syn_decay['T2S'] = self._compute_CT2S_sig(freq      = [self.MW_f['T2S'][:, None], 
                                                                                self.FW_f['T2S'][:, None]], 
                                                                   phi       =  self.phi['T2S'][:, None], 
                                                                   m_arr     = [mat1_array, mat2_array], 
                                                                   mult_CT2S = self.mult_CT2S,
                                                                   n_comp    = self.n_comp)
                    
                    if self.norm_max:
                        max_val = np.max(np.abs(self.syn_decay['T2S']), axis=1, keepdims=True)
                        max_val[max_val == 0] = 1
                        self.syn_decay['T2S'] = self.syn_decay['T2S'] / max_val

        if self.n_comp == 3:          

            if self.inv_T1:
                
                values  = np.array([self.m1['T1'], self.m2['T1'], self.m3['T1']])
                weights = np.array([(self.int2['T1']+self.int3['T1'])*self.MWF['T1']/(1 - self.MWF['T1']), self.int2['T1'], self.int3['T1']])
                widths  = np.array([self.m1_sig['T1'], self.m2_sig['T1'], self.m3_sig['T1']])
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values     = np.take_along_axis(values,  idx_sort, axis=1)
                weights    = np.take_along_axis(weights, idx_sort, axis=1)
                widths     = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2, mat_m3 = {}, {}, {}
                
                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2')
                left_m3, right_m3 = self._compute_intervall_gauss(widths[2], values[2], weights[2], signal='T2')          

                for i, idx in enumerate(self.idx_slices):        
                   
                    m1 = self._compute_gaussian(self.sys_grid['T1'][None, left_m1[i]:right_m1[i]], 
                                                values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                    m2 = self._compute_gaussian(self.sys_grid['T1'][None, left_m2[i]:right_m2[i]], 
                                                values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                    m3 = self._compute_gaussian(self.sys_grid['T1'][None, left_m3[i]:right_m3[i]], 
                                                values[2, idx, None], sig_sq[2, idx, None], fac_pre[2, idx, None])
                    
                    mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T1[left_m1[i]:right_m1[i], :])
                    mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T1[left_m2[i]:right_m2[i], :])
                    mat_m3[i] = self._compute_matmul(m3, self.fs_sysMatr_T1[left_m3[i]:right_m3[i], :])

                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                mat3_array = np.concatenate([mat_m3[k] for k in mat_m3.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                mat3_array = mat3_array[idx_unsort[2], :]
                
                self.syn_decay['T1'] = mat1_array + mat2_array + mat3_array               
            
                if self.norm_max:
                    max_val = np.max(self.syn_decay['T1'], axis=1, keepdims=True)
                    self.syn_decay['T1'] = self.syn_decay['T1']/max_val
                                                        
            if self.inv_T2:
                
                values  = np.array([self.m1['T2'], self.m2['T2'], self.m3['T2']])
                weights = np.array([(self.int2['T2']+self.int3['T2'])*self.MWF['T2']/(1 - self.MWF['T2']), self.int2['T2'], self.int3['T2']])
                widths  = np.array([self.m1_sig['T2'], self.m2_sig['T2'], self.m3_sig['T2']])
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values     = np.take_along_axis(values,  idx_sort, axis=1)
                weights    = np.take_along_axis(weights, idx_sort, axis=1)
                widths     = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2, mat_m3 = {}, {}, {}
                
                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2')
                left_m3, right_m3 = self._compute_intervall_gauss(widths[2], values[2], weights[2], signal='T2')          

                for i, idx in enumerate(self.idx_slices):        
                   
                    m1 = self._compute_gaussian(self.sys_grid['T2'][None, left_m1[i]:right_m1[i]], 
                                                values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                    m2 = self._compute_gaussian(self.sys_grid['T2'][None, left_m2[i]:right_m2[i]], 
                                                values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                    m3 = self._compute_gaussian(self.sys_grid['T2'][None, left_m3[i]:right_m3[i]], 
                                                values[2, idx, None], sig_sq[2, idx, None], fac_pre[2, idx, None])
                    
                    mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T2[left_m1[i]:right_m1[i], :])
                    mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T2[left_m2[i]:right_m2[i], :])
                    mat_m3[i] = self._compute_matmul(m3, self.fs_sysMatr_T2[left_m3[i]:right_m3[i], :])

                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                mat3_array = np.concatenate([mat_m3[k] for k in mat_m3.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                mat3_array = mat3_array[idx_unsort[2], :]
                
                self.syn_decay['T2'] = mat1_array + mat2_array + mat3_array               
            
                if self.norm_max:
                    max_val = np.max(self.syn_decay['T2'], axis=1, keepdims=True)
                    self.syn_decay['T2'] = self.syn_decay['T2']/max_val
                                                        
            if self.inv_T2S:
                
                values    = np.array([self.m1['T2S'],self.m2['T2S'],self.m3['T2S']])
                weights   = np.array([(self.int2['T2S']+self.int3['T2S'])*self.MWF['T2S']/(1-self.MWF['T2S']), self.int2['T2S'],self.int3['T2S']])
                widths    = np.array([self.m1_sig['T2S'],self.m2_sig['T2S'],self.m3_sig['T2S']])                         
                
                idx_sort   = np.argsort(values, axis=1)
                idx_unsort = np.argsort(idx_sort, axis=1)
                
                values    = np.take_along_axis(values,  idx_sort, axis=1)
                weights   = np.take_along_axis(weights, idx_sort, axis=1)
                widths    = np.take_along_axis(widths,  idx_sort, axis=1)
                
                mat_m1, mat_m2, mat_m3 = {}, {}, {}
                
                sig_sq, fac_pre   = self._compute_constants_gauss(widths, weights)
                left_m1, right_m1 = self._compute_intervall_gauss(widths[0], values[0], weights[0], signal='T2S')
                left_m2, right_m2 = self._compute_intervall_gauss(widths[1], values[1], weights[1], signal='T2S')
                left_m3, right_m3 = self._compute_intervall_gauss(widths[2], values[2], weights[2], signal='T2S')   

                for i, idx in enumerate(self.idx_slices):
           
                    m1 = self._compute_gaussian(self.sys_grid['T2S'][None, left_m1[i]:right_m1[i]], 
                                                values[0, idx, None], sig_sq[0, idx, None], fac_pre[0, idx, None])
                    m2 = self._compute_gaussian(self.sys_grid['T2S'][None, left_m2[i]:right_m2[i]], 
                                                values[1, idx, None], sig_sq[1, idx, None], fac_pre[1, idx, None])
                    m3 = self._compute_gaussian(self.sys_grid['T2S'][None, left_m3[i]:right_m3[i]], 
                                                values[2, idx, None], sig_sq[2, idx, None], fac_pre[2, idx, None])
                    
                    mat_m1[i] = self._compute_matmul(m1, self.fs_sysMatr_T2S[left_m1[i]:right_m1[i], :])
                    mat_m2[i] = self._compute_matmul(m2, self.fs_sysMatr_T2S[left_m2[i]:right_m2[i], :])
                    mat_m3[i] = self._compute_matmul(m3, self.fs_sysMatr_T2S[left_m3[i]:right_m3[i], :])
                   
                ## version 2.0.0 alpha                    
                mat1_array = np.concatenate([mat_m1[k] for k in mat_m1.keys()], axis=0)
                mat2_array = np.concatenate([mat_m2[k] for k in mat_m2.keys()], axis=0)
                mat3_array = np.concatenate([mat_m3[k] for k in mat_m3.keys()], axis=0)
                
                mat1_array = mat1_array[idx_unsort[0], :]
                mat2_array = mat2_array[idx_unsort[1], :]
                mat3_array = mat3_array[idx_unsort[2], :]
                    
                if not self.inv_CT2S:                        
                    self.syn_decay['T2S'] = mat1_array + mat2_array + mat3_array
                    
                    if self.norm_max:
                        max_val = np.max(self.syn_decay['T2S'], axis=1, keepdims=True)
                        self.syn_decay['T2S'] = self.syn_decay['T2S']/max_val
                
                else:
                    self.syn_decay['T2S'] = self._compute_CT2S_sig(freq      = [self.MW_f['T2S'][:, None], 
                                                                                self.EW_f['T2S'][:, None],
                                                                                self.AW_f['T2S'][:, None]],
                                                                   phi       =  self.phi['T2S'][:, None], 
                                                                   m_arr     = [mat1_array, mat2_array, mat3_array], 
                                                                   mult_CT2S = self.mult_CT2S,
                                                                   n_comp    = self.n_comp)
                    
                    if self.norm_max:
                        max_val = np.max(np.abs(self.syn_decay['T2S']), axis=1, keepdims=True)
                        max_val[max_val == 0] = 1
                        self.syn_decay['T2S'] = self.syn_decay['T2S'] / max_val                    
                        
###############################################################################

    def fitness(self):
        
        """Computes particle fitness values based on modelâ€“observation misfit."""
        
        # NOTE for CT2S signal: phase and magnitude could also be weighted differently 
        #
        # NOTE lp-norm: L1 norm is not fully implemented yet.         
        
        if self.inv_SI==True:
                
            sig = self.decay_types[0]
         
            if self.lp_norm == 'L1':
                self.fit[sig] = np.sum(np.abs(self.obs_decay[sig]-self.syn_decay[sig]), axis=1)/len(self.obs_decay[sig])
    
            if self.lp_norm == 'L2': # Frobenius norm:
                if self.inv_CT2S==False:
                    self.fit[sig] = np.sqrt(np.sum((self.obs_decay[sig]-self.syn_decay[sig])**2, axis=1))/len(self.obs_decay[sig])
                
                if self.inv_CT2S==True:
                    
                    # self.fit[sig] = np.sqrt(np.sum((np.abs(self.obs_decay[sig])-np.abs(self.syn_decay[sig]))**2, axis=1))/len(self.obs_decay[sig]) + \
                    #                 np.sqrt(np.sum((np.angle(self.obs_decay[sig])-np.angle(self.syn_decay[sig]))**2, axis=1))/len(self.obs_decay[sig])
                    
                    self.fit[sig] = np.sqrt(np.sum(np.abs(self.obs_decay[sig]-self.syn_decay[sig])**2, axis=1))/len(self.obs_decay[sig])
                
        if self.inv_JI==True:
            
            alpha, beta, gamma = self.weights[0], self.weights[1], self.weights[2]
            
            term        = {sig:[] for sig in self.decay_types}
            term['MWF'] = []
            keys        = list(term.keys())                # e.g. T2, T2S & MWF
            
            for sig in self.decay_types:

                term[sig] = np.sqrt(np.sum((self.obs_decay[sig]-self.syn_decay[sig])**2, axis=1))/len(self.obs_decay[sig])
                
                if sig == 'T2S' and self.inv_CT2S == True:
                    
                    ### complexe difference needs no unwrapping of the phase component
                    term['T2S'] = np.sqrt(np.sum(np.abs(self.obs_decay[sig]-self.syn_decay[sig])**2, axis=1))/len(self.obs_decay[sig])
                    
                    ### component based difference needs unwrapping of the phase component, but is prone to noise
                    # term['T2S'] = np.sqrt(np.sum((np.abs(self.obs_decay[sig])-np.abs(self.syn_decay[sig]))**2, axis=1))/len(self.obs_decay[sig]) + \
                                  # np.sqrt(np.sum((np.unwrap(np.angle(self.obs_decay[sig]))-np.unwrap(np.angle(self.syn_decay[sig])))**2, axis=1))/len(self.obs_decay[sig])
 
            term['MWF'] = (self.MWF[keys[0]]-self.MWF[keys[1]])**2
            
            self.fit[self.decay_JI] = alpha*term[keys[0]] + beta*term[keys[1]] + gamma*term[keys[2]]    

###############################################################################

    def mean_misfit(self, data1: np.array, data2: np.array):
        
        """Computes mean misfit between two data arrays using the selected lp-norm."""

        if self.lp_norm == 'L1':
            return np.sum(np.abs(data1-data2))/len(data1)
    
        if self.lp_norm == 'L2':
            return np.sqrt(np.sum((data1-data2)**2)/len(data1))            
        
###############################################################################            
    
    def best_local(self):

        """Updates particle-wise personal best solutions (local bests)."""

        if self.inv_SI == True:
            
            sig  = self.decay_types[0]
    
            fit  = self.fit[sig].copy()
            bfit = self.best_fit[sig].copy()
            mask = fit <= bfit
            
            self.best_fit[sig][mask]      = fit[mask]
            self.best_MWF[sig][mask]      = self.MWF[sig][mask].copy()
            self.best_mod[sig][mask]      = self.mod[sig][mask].copy()
            self.best_syn_data[sig][mask] = self.syn_decay[sig][mask].copy()

        if self.inv_JI == True:   

            i,j,k = self.decay_types[0], self.decay_types[1], self.decay_JI
            fit   = self.fit[k].copy()
            bfit  = self.best_fit[k].copy()
            mask  = fit <= bfit
            
            for sig in self.decay_types:

                self.best_MWF[sig][mask]      = self.MWF[sig][mask].copy()
                self.best_mod[sig][mask]      = self.mod[sig][mask].copy()
                self.best_syn_data[sig][mask] = self.syn_decay[sig][mask].copy()
                
            self.best_fit[k][mask] = fit[mask]
            self.best_MWF[k]       = np.sum([self.best_MWF[i],self.best_MWF[j]], axis=0)/len(self.decay_types)

###############################################################################

    def best_global(self):

        """
        Updates the global best position among all particles.
        --> drives convergence toward the swarmâ€™s optimal solution
        """

        if self.inv_SI == True:
            
            sig       = self.decay_types[0]
            fit       = self.best_fit[sig]
            idx_min   = np.argmin(fit)
            val_min   = fit[idx_min]
        
            if val_min < self.glob_fit[sig]:
                self.glob_fit[sig]      = val_min
                self.glob_mod[sig]      = np.copy(self.mod[sig][idx_min])
                self.glob_syn_data[sig] = np.copy(self.best_syn_data[sig][idx_min])
                self.glob_ind[sig]      = idx_min
        
        if self.inv_JI == True:
            
            k         = self.decay_JI
            fit       = self.best_fit[k].copy()
            idx_min   = np.argmin(fit)
            val_min   = fit[idx_min]

            if val_min < self.glob_fit[k]:
                self.glob_fit[k] = val_min
                self.glob_ind[k] = idx_min
                
                for sig in self.decay_types:
                    self.glob_syn_data[sig] = np.copy(self.best_syn_data[sig][idx_min])
                    self.glob_mod[sig]      = np.copy(self.mod[sig][idx_min])
        
###############################################################################

    def update_position(self):
        
        '''
        Updates particle positions within the multidimensional search space.
        --> position update governed by inertia, cognitive, and social components
        '''
        
        w  = self.PSO.w                               # inertia weight factor
        c1 = self.PSO.c1                              # social weight factor
        c2 = self.PSO.c2                              # cognitive weight factor

        for sig in self.decay_types:

            r = np.random.rand(self.n_part*2, self.n_param[sig])
            r11, r22 = r[::2], r[1::2]
            
            self.vel[sig] = w*self.vel[sig] + \
                            c1*r11*(self.best_mod[sig]-self.mod[sig]) + \
                            c2*r22*(self.glob_mod[sig]-self.mod[sig])
            
            self.mod[sig] = self.mod[sig] + self.vel[sig]
            
###############################################################################
    
    def check_limit(self):

        '''
        Enforces parameter boundaries for all particles.
        --> uses np.clip() for efficient constraint handling
        '''
        
        for sig in self.decay_types:
            
            if self.n_comp == 2:

                # Clip core Gaussian parameters (applies to T1, T2 or T2S/CT2S)
                self.m1[sig]     = np.clip(self.mod[sig][:,0], self.att[sig].m1[0],     self.att[sig].m1[1])
                self.m1_sig[sig] = np.clip(self.mod[sig][:,1], self.att[sig].m1_sig[0], self.att[sig].m1_sig[1])              
                self.m2[sig]     = np.clip(self.mod[sig][:,2], self.att[sig].m2[0],     self.att[sig].m2[1])
                self.m2_sig[sig] = np.clip(self.mod[sig][:,3], self.att[sig].m2_sig[0], self.att[sig].m2_sig[1])
                self.int2[sig]   = np.clip(self.mod[sig][:,4], self.att[sig].int2[0],   self.att[sig].int2[1])
                self.MWF[sig]    = np.clip(self.mod[sig][:,5], self.att[sig].MWF[0],    self.att[sig].MWF[1])
                
                # Case 1: Real-valued T2 or real-valued T2S (no CT2S model)
                if sig == 'T1' or sig == 'T2' or (sig == 'T2S' and not self.inv_CT2S):
                  
                    self.vel[sig] = np.where(np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig], 
                                                       self.m2_sig[sig], self.int2[sig],   self.MWF[sig]], 
                                                       axis=1) != self.mod[sig], 0, self.vel[sig])
            
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   
                                              self.m2_sig[sig], self.int2[sig],   self.MWF[sig]], axis=1)
                
                # Case 2: complex T2S with CT2S (frequency shifts + phase)
                if sig == 'T2S' and self.inv_CT2S:
                    
                    self.MW_f[sig] = np.clip(self.mod[sig][:,6], self.att[sig].MW_f[0], self.att[sig].MW_f[1])
                    self.FW_f[sig] = np.clip(self.mod[sig][:,7], self.att[sig].FW_f[0], self.att[sig].FW_f[1])
                    self.phi[sig]  = np.clip(self.mod[sig][:,8], self.att[sig].phi[0],  self.att[sig].phi[1])
                    
                    self.vel[sig] = np.where(np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig], 
                                                       self.m2_sig[sig], self.int2[sig],   self.MWF[sig],
                                                       self.MW_f[sig],   self.FW_f[sig],   self.phi[sig]],
                                                       axis=1) != self.mod[sig], 0, self.vel[sig])                    

                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   
                                              self.m2_sig[sig], self.int2[sig],   self.MWF[sig], 
                                              self.MW_f[sig],   self.FW_f[sig],   self.phi[sig]],  axis=1)
                    
        
            elif self.n_comp == 3:
 
                # Clip core Gaussian parameters (applies to T1, T2 or T2S/CT2S)
                self.m1[sig]     = np.clip(self.mod[sig][:,0], self.att[sig].m1[0],      self.att[sig].m1[1])
                self.m1_sig[sig] = np.clip(self.mod[sig][:,1], self.att[sig].m1_sig[0],  self.att[sig].m1_sig[1])              
                self.m2[sig]     = np.clip(self.mod[sig][:,2], self.att[sig].m2[0],      self.att[sig].m2[1])
                self.m2_sig[sig] = np.clip(self.mod[sig][:,3], self.att[sig].m2_sig[0],  self.att[sig].m2_sig[1])
                self.m3[sig]     = np.clip(self.mod[sig][:,4], self.att[sig].m3[0],      self.att[sig].m3[1])
                self.m3_sig[sig] = np.clip(self.mod[sig][:,5], self.att[sig].m3_sig[0],  self.att[sig].m3_sig[1])
                self.int2[sig]   = np.clip(self.mod[sig][:,6], self.att[sig].int2[0],    self.att[sig].int2[1])
                self.int3[sig]   = np.clip(self.mod[sig][:,7], self.att[sig].int3[0],    self.att[sig].int3[1])
                self.MWF[sig]    = np.clip(self.mod[sig][:,8], self.att[sig].MWF[0],     self.att[sig].MWF[1])
                
                # Case 1: Real-valued T2 or real-valued T2S (no CT2S model)
                if sig == 'T1' or sig == 'T2' or (sig == 'T2S' and not self.inv_CT2S):
                
                    self.vel[sig] = np.where(np.stack([self.m1[sig], self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig],
                                                       self.m3[sig], self.m3_sig[sig], self.int2[sig], self.int3[sig], 
                                                       self.MWF[sig]], axis=1) != self.mod[sig], 0, self.vel[sig])
                    
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig], self.m3[sig], 
                                              self.m3_sig[sig], self.int2[sig],   self.int3[sig], self.MWF[sig]],   axis=1)                              

                # Case 2: complex T2S with CT2S (frequency shifts + phase)
                if sig == 'T2S' and self.inv_CT2S:
                
                    self.MW_f[sig] = np.clip(self.mod[sig][:,9],  self.att[sig].MW_f[0], self.att[sig].MW_f[1])
                    self.EW_f[sig] = np.clip(self.mod[sig][:,10], self.att[sig].EW_f[0], self.att[sig].EW_f[1])
                    self.AW_f[sig] = np.clip(self.mod[sig][:,11], self.att[sig].AW_f[0], self.att[sig].AW_f[1])                   
                    self.phi[sig]  = np.clip(self.mod[sig][:,12], self.att[sig].phi[0],  self.att[sig].phi[1])
                        
                    self.vel[sig] = np.where(np.stack([self.m1[sig],   self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig],
                                                       self.m3[sig],   self.m3_sig[sig], self.int2[sig], self.int3[sig], 
                                                       self.MWF[sig],  self.MW_f[sig],   self.EW_f[sig], self.AW_f[sig], 
                                                       self.phi[sig]], axis=1) != self.mod[sig], 0, self.vel[sig])
                    
                    self.mod[sig] = np.stack([self.m1[sig],     self.m1_sig[sig], self.m2[sig],   self.m2_sig[sig], self.m3[sig], 
                                              self.m3_sig[sig], self.int2[sig],   self.int3[sig], self.MWF[sig],    self.MW_f[sig], 
                                              self.EW_f[sig],   self.AW_f[sig],   self.phi[sig]],  axis=1)
                    

###############################################################################

    def init_system_matrix(self, root_MWF: dict(), **kwargs):
        
        """Initialize system matrices for enabled MRI signals."""
        
        yy, xx = kwargs.get('position', (0,0))
        
        self.sys_matrix['T1']  = root_MWF['T1_MATRIX']  if self.inv_T1==True else []
        self.sys_matrix['T2']  = root_MWF['T2_MATRIX']  if self.inv_T2==True else []
        self.sys_matrix['T2S'] = root_MWF['T2S_MATRIX'] if self.inv_T2S==True else []
        
###############################################################################
    
    def init_grid(self):
        
        """Initialize system grids for all supported MRI signals."""
        
        self.sys_grid['T1']    = hlp.make_grid(self.Inv.T1_min, 
                                               self.Inv.T1_max,
                                               self.Inv.mod_space, mode='lin')

        self.sys_grid['T2']    = hlp.make_grid(self.Inv.T2_min, 
                                               self.Inv.T2_max,
                                               self.Inv.mod_space, mode='lin')

        self.sys_grid['T2S']   = hlp.make_grid(self.Inv.T2S_min, 
                                               self.Inv.T2S_max,
                                               self.Inv.mod_space, mode='lin')
    
###############################################################################    
        
    def init_particle_swarm(self):
        
        """Initialize the particle swarm for the first PSO iteration."""
        
        # Generate synthetic data based on the current particle model parameters
        self.compute_synthetic_decay()
        
        # Compute particle fitness values based on modelâ€“observation misfit
        self.fitness()

        # Update particle-wise personal best solutions (local bests)
        self.best_local()

        # Update the global best position among all particles
        self.best_global()
        
        # Collect global best parameters for the very first iteration
        if self.plot_iter_test and self.inv_SI:            
            sig                     = self.decay_types[0]
            self.glob_ind_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.glob_fit_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.glob_MWF_list      = {'T1': [], 'T2': [], 'T2S': []}
            self.glob_ind_list[sig] = [self.glob_ind[sig]+1]
            self.glob_fit_list[sig] = [self.glob_fit[sig]]
            self.glob_MWF_list[sig] = [self.glob_mod[sig][-1]]
        
###############################################################################  
    
    def run_pso(self, obs_decay: np.array, plot_iter_test=False, callback_plot=None):
        
        """Execute the particle swarm optimization loop."""

        self.plot_iter_test = plot_iter_test
        self.obs_decay      = obs_decay

        # Initalization of the particle swarm
        self.init_particle_swarm()
        
        for jj in range(self.n_iter)[:]:

            # Update particle positions within the multidimensional search space
            # Each Particle  -->  locVector = w*v1 + c1*r1*(m[i+1]-m[i])              
            self.update_position()
            
            # Enforce parameter boundaries for all particles     
            self.check_limit()
    
            # Update synthetic data based on the current particle model parameters
            self.compute_synthetic_decay()
                        
            # Compute particle fitness values based on modelâ€“observation misfit
            self.fitness()
            
            # Update particle-wise personal best solutions (local bests)
            self.best_local()
      
            # Update the global best position among all particles
            self.best_global()

            # Collect global best parameters for a number of n iterations
            if self.plot_iter_test and self.inv_SI:
                sig = self.decay_types[0]
                self.glob_ind_list[sig].append(self.glob_ind[sig]+1)
                self.glob_fit_list[sig].append(self.glob_fit[sig])
                self.glob_MWF_list[sig].append(self.glob_mod[sig][-1])

                # if callback_plot is not None and jj in [1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99]:
                #     string = [self.decay_types[0], str(jj).zfill(3)]
                #     callback_plot(PSO=self, string=string)

###############################################################################

    def write_csv(self):
        """Write PSO results to CSV format (not implemented yet)."""
        return

###############################################################################
    
    def write_json(self, path: str):
        
        """Write PSO configuration and model parameters to a JSON file."""

        save_path    = os.path.join(path, 'parameters.json')                                    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        if self.inv_SI==True:
          
            sig       = self.decay_types[0]
            sig_label = 'CT2S' if (self.inv_CT2S and sig=='T2S') else sig
            dat_space = getattr(self.Inv, f"n_echoes_{sig}")
    
            json_data    = {"data source":{
                                "_signal":      sig_label,
                                "_source":      self.data_source,
                                "_brain regions": None,
                                "_noise added": False,
                                "_norm decay":  self.config.source.data.norm_max,
                                "_SNR":         np.nan,
                                "_n echos":     dat_space},
                           "model vector parameters":{
                                "_m1":          getattr(getattr(self, sig), self.param_class).m1,
                                "_m1_sig":      getattr(getattr(self, sig), self.param_class).m1_sig, 
                                "_m2":          getattr(getattr(self, sig), self.param_class).m2,
                                "_m2_sig":      getattr(getattr(self, sig), self.param_class).m2_sig, 
                                "_m3":          None,
                                "_m3_sig":      None,
                                "_int2":        getattr(getattr(self, sig), self.param_class).int2,
                                "_int3":        None,
                                "_MWF":         getattr(getattr(self, sig), self.param_class).MWF,
                                "_MW_f":        None,
                                "_FW_f":        None,
                                "_EW_f":        None,
                                "_AW_f":        None,
                                "_phi":         None},
                           "PSO specifications":{
                                "_PSO version": self.config.general.PSO_version,
                                "_components":  self.n_comp,
                                "_iterations":  self.n_iter,
                                "_particles":   self.n_part,
                                "_PSO cycles":  self.n_pso_cycles,
                                "_w":           self.PSO.w,
                                "_c1":          self.PSO.c1,
                                "_c2":          self.PSO.c2,
                                "_LP Norm":     self.lp_norm,
                                "_weights":     self.weights,
                                "_norm decay":  self.norm_max,
                                "_random seed": self.config.PSO_spec.PSO_math.rand},}
            
            if self.data_source == 'invivo':
                json_data["data source"]["_brain regions"] = self.brain_regions
                
            if self.n_comp == 3:
                json_data["model vector parameters"]["_m3"]     = getattr(getattr(self, sig), self.param_class).m3
                json_data["model vector parameters"]["_m3_sig"] = getattr(getattr(self, sig), self.param_class).m3_sig
                json_data["model vector parameters"]["_int3"]   = getattr(getattr(self, sig), self.param_class).int3
                
                if self.inv_CT2S and sig == 'T2S':
                    json_data["model vector parameters"]["_MW_f"] = getattr(getattr(self, sig), self.param_class).MW_f
                    json_data["model vector parameters"]["_EW_f"] = getattr(getattr(self, sig), self.param_class).EW_f
                    json_data["model vector parameters"]["_AW_f"] = getattr(getattr(self, sig), self.param_class).AW_f
                    json_data["model vector parameters"]["_phi"]  = getattr(getattr(self, sig), self.param_class).phi
                    
            if self.n_comp == 2 and self.inv_CT2S:
                json_data["model vector parameters"]["_MW_f"] = getattr(getattr(self, sig), self.param_class).MW_f
                json_data["model vector parameters"]["_FW_f"] = getattr(getattr(self, sig), self.param_class).FW_f
                json_data["model vector parameters"]["_phi"]  = getattr(getattr(self, sig), self.param_class).phi
            
            if self.add_noise:
                json_data["data source"]["_noise added"] = True 
                json_data["data source"]["_SNR"]         = self.SNR
                
            mod_vec_param = json_data["model vector parameters"]
            mod_vec_param = {k: v for k, v in mod_vec_param.items() if v is not None}
            json_data["model vector parameters"] = mod_vec_param
    
            f = json.dumps(json_data).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                            
            with open(save_path, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
        if self.inv_JI==True:
            
            sig1, sig2     = self.decay_types[0], self.decay_types[1]
            dat_space_sig1 = getattr(self.Inv, f"n_echoes_{sig1}")
            dat_space_sig2 = getattr(self.Inv, f"n_echoes_{sig2}")
            
            json_data    = {"data source":{
                                "_signal":      self.decay_JI,
                                "_source":      self.data_source,
                                "_brain regions": None,
                                "_noise added": False,
                                "_norm decay":  self.config.source.data.norm_max,
                                "_SNR":         np.nan,
                                "_n echos T2":  dat_space_sig1,
                                "_n echos T2S": dat_space_sig2},
                           f"model parameters {sig1}":{
                                "_m1":          getattr(getattr(self, sig1), self.param_class).m1,
                                "_m1_sig":      getattr(getattr(self, sig1), self.param_class).m1_sig, 
                                "_m2":          getattr(getattr(self, sig1), self.param_class).m2,
                                "_m2_sig":      getattr(getattr(self, sig1), self.param_class).m2_sig, 
                                "_m3":          None,
                                "_m3_sig":      None,
                                "_int2":        getattr(getattr(self, sig1), self.param_class).int2,
                                "_int3":        None,
                                "_MWF":         getattr(getattr(self, sig1), self.param_class).MWF},
                           f"model parameters {sig2}":{
                                "_m1":          getattr(getattr(self, sig2), self.param_class).m1,
                                "_m1_sig":      getattr(getattr(self, sig2), self.param_class).m1_sig, 
                                "_m2":          getattr(getattr(self, sig2), self.param_class).m2,
                                "_m2_sig":      getattr(getattr(self, sig2), self.param_class).m2_sig, 
                                "_m3":          None,
                                "_m3_sig":      None,
                                "_integ2":      getattr(getattr(self, sig2), self.param_class).int2,
                                "_integ3":      None,
                                "_MWF":         getattr(getattr(self, sig2), self.param_class).MWF,
                                "_MW_f":        None,
                                "_FW_f":        None,
                                "_EW_f":        None,
                                "_AW_f":        None,
                                "_phi":         None},
                           "PSO specifications":{
                                "_PSO version": self.config.general.PSO_version,
                                "_components":  self.n_comp,
                                "_iterations":  self.n_iter,
                                "_particles":   self.n_part,
                                "_PSO cycles":  self.n_pso_cycles,
                                "_w":           self.PSO.w,
                                "_c1":          self.PSO.c1,
                                "_c2":          self.PSO.c2,
                                "_LP Norm":     self.lp_norm,
                                "_weights":     self.weights,
                                "_norm decay":  self.norm_max,
                                "_random seed": self.config.PSO_spec.PSO_math.rand},}

            if self.data_source == 'invivo':
                json_data["data source"]["_brain regions"] = self.brain_regions
                
            if self.n_comp == 3:
                for sig in self.decay_types:                    
                    json_data[f"model parameters {sig}"]["_m3"]     = getattr(getattr(self, sig), self.param_class).m3
                    json_data[f"model parameters {sig}"]["_m3_sig"] = getattr(getattr(self, sig), self.param_class).m3_sig
                    json_data[f"model parameters {sig}"]["_int  3"] = getattr(getattr(self, sig), self.param_class).int3
                
                if self.inv_CT2S:
                    json_data["model parameters T2S"]["_MW_f"] = getattr(getattr(self, 'T2S'), self.param_class).MW_f
                    json_data["model parameters T2S"]["_EW_f"] = getattr(getattr(self, 'T2S'), self.param_class).EW_f
                    json_data["model parameters T2S"]["_AW_f"] = getattr(getattr(self, 'T2S'), self.param_class).AW_f
                    json_data["model parameters T2S"]["_phi"]  = getattr(getattr(self, 'T2S'), self.param_class).phi
               
            if self.n_comp == 2:
                for sig in self.decay_types:  
                    syn_data = json_data[f"model parameters {sig}"]
                    syn_data = {k: v for k, v in syn_data.items() if v is not None}
                    json_data[f"model parameters {sig}"] = syn_data
            
                if self.inv_CT2S:
                    json_data["model parameters T2S"]["_MW_f"] = getattr(getattr(self, 'T2S'), self.param_class).MW_f
                    json_data["model parameters T2S"]["_FW_f"] = getattr(getattr(self, 'T2S'), self.param_class).FW_f
                    json_data["model parameters T2S"]["_phi"]  = getattr(getattr(self, 'T2S'), self.param_class).phi
            
            if self.add_noise:
                json_data["data source"]["_noise added"] = True 
                json_data["data source"]["_SNR"]         = self.SNR

            f = json.dumps(json_data).replace(', "', '; "').replace('"_', '\t"')
            f = f.replace('; ', ',\n').replace('{', '{\n').replace('}','\n}')
                        
            with open(save_path, 'w') as json_file:
                json_file.write(f) # without any string-changes use json.dump():
                                   # json.dump(data, json_file, indent='\t', separators=(',', ':'))
        
###############################################################################
    
    def dict_to_array(self, results: dict, result_map: np.array, 
                     syn_dat_map: np.array, res_array_dic: np.array, kk: int, **kwargs):
        
        '''
        Converts results from the PSO results dictionary into numpy array
        objects for each inverted decay type.        
        '''

        calcBestResult = kwargs.get('calcBestResult', False)
        arrayType      = kwargs.get('arrayType', 'Slice')
        
        results_ = [i for i in results if i != None]

        if arrayType == 'Pixel':
            
            for sig in self.decay_types:
                
                for i, line in enumerate(results_):
                    result_map[sig][:-1, i] = line[f'mod{sig}']
                    result_map[sig][-1,  i] = line['fit']
        
        if arrayType == 'Slice':
            
            for sig in self.decay_types:
                
                if self.inv_CT2S == True:
                    syn_dat_map[sig] = syn_dat_map[sig].astype(np.complex128)
                    
        #######################################################################
                # synthetic data save
                for item in results:
                    y, x = item['pix']
                    
                    syn_dat_map[sig][y, x, :, kk] = item[f'syn_data{sig}']
                    res_array_dic[y, x, kk]       = item

        ####################################################################### 
                for line in results_:
                    yy,xx = line['pix'][0],line['pix'][1]
                    
                    result_map[sig][yy,xx,-1,kk] = line['fit']
            
                    for ii,item in enumerate(line[f'mod{sig}']):
                        result_map[sig][yy,xx,ii,kk] = item        
            
                    if kk == self.n_pso_cycles-1 and calcBestResult==True:
                        result_map[sig] = self.__bestfit2array__(result_map[sig],
                                                                 position=(yy,xx))

                
                # for line in results_:
                #     yy,xx = line['pix'][0],line['pix'][1]
                    
                #     result_map[sig][-1,yy,xx,kk] = line['fit']
            
                #     for ii,item in enumerate(line[f'mod{sig}']):
                #         result_map[sig][ii,yy,xx,kk] = item        
            
                #     if kk == self.n_pso_cycles-1 and calcBestResult==True:
                #         result_map[sig] = self.__bestfit2array__(result_map[sig],
                #                                                  position=(yy,xx))

        return result_map, syn_dat_map, res_array_dic
        
###############################################################################

    # def __bestfit2array__(self, result_map: np.array, position: tuple):
        
    #     yy,xx = position[0], position[1]
        
    #     try:
    #         bestFitID              = np.nanargmin(result_map[-1,yy,xx,:-1])
    #         result_map[:,yy,xx,-1] = result_map[:,yy,xx,bestFitID]
    #     except:
    #         result_map[:,yy,xx,-1] = np.nan
        
    #     return result_map

    def __bestfit2array__(self, result_map: np.array, position: tuple):
        
        yy,xx = position[0], position[1]
        
        try:
            bestFitID              = np.nanargmin(result_map[yy,xx,-1,:-1])
            result_map[yy,xx,:,-1] = result_map[yy,xx,:,bestFitID]
        except:
            result_map[yy,xx,:,-1] = np.nan
        
        return result_map

###############################################################################

    def log(self, start_time, string='', dim='sek', boolean=False):
        
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

        t_now       = time.time()
        t_elapsed   = t_now - start_time
        
        if dim=='HMS':
            tt = time.strftime('%H:%M:%S', time.gmtime(t_elapsed))
            print(f'{string}: {tt} hrs')
            
        if dim=='MS':
            tt = time.strftime('%M:%S', time.gmtime(t_elapsed))
            print(f'{string}: {tt} min')
        
        if dim=='ms':
            t_ms        = round(t_elapsed*1000, 2)
            
            if t_ms >0:
                print(f'{string}: {t_ms} ms')
        
        if dim=='mus':
            t_mus        = round(t_elapsed*1000 * 1000, 2)
            print(f'{string}: {t_mus} mus')#'\u03BCs')