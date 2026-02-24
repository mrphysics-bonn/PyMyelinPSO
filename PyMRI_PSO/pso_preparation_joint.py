# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# Author:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of the PyMRI_PSO software.
# See the LICENSE file in the project root for full license information.

"""
Side script for preparing particle swarm optimization (PSO) on MRI invivo data.
Inversion type: single inversion.

Purpose:
    a) Generates and/or fills arrays with observed MRI data both from MRI invivo
       and/or a published MRI atlas (Dvorak et al., 1021).
    b) Loads or generates binary masks used to restrict the PSO analysis.
    c1) Prepares and computes the full system model cube.
    c2) Data pre-processing, B1 handling, model setup (T1, T2, T2S).
    d) Fills the system parameters dictionary used in the PSO analysis for one 
       or multiple data cube slices of interest.
"""

import os, time, copy
import numpy   as     np
import nibabel as     nib
from   pathlib import Path

import help_tools   as     hlp
from   mwf_modeling import mwf_analysis
#from   mwf_t1t2t2s  import mwf_analysis
from   pso_core     import ParticleSwarmOptimizer as PSOclass


class PSOpreparation_JI(PSOclass):

    def __init__(self, config_input: str, config_type: str, mask_use=None, 
                 mask_path=['',''], data_type='invivo', verbose=False):
        
        """
        Initializes the preparation of PSO based on joint-inversion.
        Loads configuration, sets paths and mask parameters, and inherits PSO methods.
        
        *args:
            config_input : configuration file path or dict
            config_type  : 'file' or 'dict'
            mask_use     : mask handling mode (bool or str)
            mask_path    : optional mask file path
            data_type    : 'invivo' or 'atlas'
            verbose      : verbose output for debugging and progress monitoring
        """

        # Load configuration depending on input type
        if config_type == 'file':
            self.config_path = config_input
            self.config2object()
        else:
            self.config      = config_input
        
        # Store general parameters
        self.data_type = data_type
        self.proj_dir  = self.config['source']['file']['prj_dir']
        self.data_dir  = os.path.join(self.proj_dir, 'nifti')
        
        # Mask handling parameters
        self.mask_use  = mask_use
        self.mask_path = mask_path
        
        # additional printout in the console
        self.verbose   = verbose
        
        # Inherit PSO base class functionality
        super().__init__(self.config)
    
    def config2object(self):
        
        """Opens and parses configuration file into a Python dictionary."""
        
        self.config = hlp.open_parameters(path=self.config_path)      
    
    def handle_mask(self):
        
        """
        Loads or generates the binary masks used to restrict PSO analysis.
        Depending on configuration, existing masks are loaded from disk
        or full-size masks of ones are created.
        """

        if self.data_type == 'atlas':
            return
                
        if self.mask_use == False:
            self.masks   = {sig: np.ones(self.obs_data[sig].shape[:3],dtype=int) for sig in self.decay_types}
            
        if self.mask_use == True:
            mask_suffix    = {sig: Path(self.mask_path[sig]).suffixes[0] for sig in self.decay_types}
            self.mask_path = {sig: os.path.join(self.proj_dir, self.mask_path[sig]) for sig in self.decay_types}
            self.masks     = {sig: [] for sig in self.decay_types}
            
            for sig in mask_suffix.keys():
                
                if mask_suffix[sig] == '.nii':
                    mask_dummy      = nib.load(self.mask_path[sig])
                    self.masks[sig] = mask_dummy.get_fdata()
                    self.masks_raw  = copy.deepcopy(self.masks)
                    self.mask_roi   = self.config.source.mask.seg
                    self.mask_roi   = [x - 1 for x in self.mask_roi]
                    
                    self.masks[sig] = ((self.masks[sig] >= np.min(self.mask_roi)) &
                                       (self.masks[sig] <= np.max(self.mask_roi))).astype(int)
                
                if mask_suffix == '.npy':
                    self.masks[sig] = np.load(file=self.mask_path[sig],allow_pickle=True)
                
    
    def access_mwf_analysis(self):
        
        """
        Initializes the MWF analysis object (mwf_analysis) depending on data type.
        Links PSO preparation to MRI data and metadata by loading relevant sources.
        """
        
        if self.data_type == 'atlas':
            self.root_mwf = mwf_analysis(data_dir = self.data_dir, 
                                         T2S_TE   = self.config.source.file.TE)
        
        if self.data_type == 'invivo':
            self.root_mwf = mwf_analysis(data_dir = self.data_dir,
                                         KW_B1    = self.config.source.file.B1,
                                         #KW_T1    = self.config.source.file.T1,
                                         KW_T2    = self.config.source.file.T2,
                                         KW_T2S   = self.config.source.file.T2S,
                                         KW_T2SP  = self.config.source.file.CT2S,
                                         T2S_TE   = self.config.source.file.TE)


    def calc_system_cube(self, slice_num):
        
        """
        Prepares and computes the full system model cube for one slice (slice_num).
        Includes data preprocessing, model setup (T1, T2, T2S), and B1 handling.
        """
        
        if self.data_type == 'atlas':
            # No preprocessing applied — ignores B1 field inhomogeneity
            pass
        
        if self.data_type == 'invivo':
            
            self.b1_grid = {sig: [] for sig in self.decay_types}
            
            for sig in self.decay_types:
                
                sig_used = 'CT2S' if sig == 'T2S' and self.config.source.signal.CT2S else sig
                
                self.root_mwf.prep_data(axis        = 'Z',
                                        slice_num   = slice_num,
                                        signal_type = sig_used,
                                        filter      = 0.0,
                                        thresh      = 4.5,
                                        verbose     = False)
                    
                self.b1_grid[sig] = self.root_mwf.b1_grid if sig == 'T2' else np.zeros((1,))
                
                if self.mask_use == False:
                    self.masks[sig][:, :, slice_num] = self.root_mwf.data.msk

        if self.inv_T2 == True:
            
            if self.data_type == 'invivo':
                self.root_mwf.prep_t2_model(te      = self.Inv.T2_TE,         # 6.6  | sego: 6.0
                                            tr      = self.Inv.T2_TR,         # 900  | sego: 2000
                                            etl     = self.Inv.n_echoes_T2,   # 24   | sego: 24
                                            alpha   = self.Inv.T2_alpha,      # 70   | sego: 90
                                            beta    = self.Inv.T2_beta,       # 180  | sego: 160
                                            T2min   = self.Inv.T2_min,        # 1    | sego: 1
                                            T2max   = self.Inv.T2_max,        # 200  | sego  200
                                            nT2     = self.Inv.mod_space,     # 1000 | sego: 1000
                                            T1      = self.Inv.T2_T1,         # 1000 | sego: 1000
                                            verbose = False)
            
            elif self.data_type == 'atlas':
                self.root_mwf.prep_t2_model(te      = 6.0,                    # 6.6  | sego: 6.0
                                            tr      = 2000,                   # 900  | sego: 2000
                                            etl     = self.Inv.n_echoes_T2,   # 24   | sego: 24
                                            alpha   = 90,                     # 70   | sego: 90
                                            beta    = 160,                    # 180  | sego: 160
                                            T2min   = self.Inv.T2_min,        # 1    | sego: 1
                                            T2max   = self.Inv.T2_max,        # 200  | sego  200
                                            nT2     = self.Inv.mod_space,     # 1000 | sego: 1000
                                            T1      = self.Inv.T2_T1,         # 1000 | sego: 1000
                                            verbose = False)
                
        if self.inv_T2S == True:
            self.root_mwf.prep_t2s_model(T2Smin  = self.Inv.T2S_min,
                                         T2Smax  = self.Inv.T2S_max,
                                         nT2S    = self.Inv.mod_space,
                                         verbose = False)
    
        if self.inv_T1 == True:                
            self.root_mwf.prep_t1_model(tr      = self.Inv.T1_TR,
                                        alpha   = self.Inv.T1_alpha,
                                        td      = self.Inv.T1_TD,
                                        ie      = self.Inv.T1_IE,
                                        T1min   = self.Inv.T1_min,
                                        T1max   = self.Inv.T1_max,
                                        nT1     = self.Inv.mod_space,
                                        verbose = False)
        
    def get_atlas_data(self):        
        
        """Loads MWF atlas from a given file and returns it as a NumPy array."""
        
        mwf_mean  = nib.load(os.path.join(self.data_dir, self.config.source.file.ATLAS))
        mwf_data  = mwf_mean.get_fdata()
        
        return mwf_data
    
    def get_obs_data(self, _data_type='invivo'):
        
        """Loads and pre-processes observed MRI measurment data from file."""
        
        self.obs_data = {'T1':[],'T2':[],'T2S':[], 'TE':[],'MWF_ATLAS':[]}
        self.norm_max = self.config.source.data.norm_max
        self.nifti_affine_T1   = None
        self.nifti_affine_T2   = None
        self.nifti_affine_T2S  = None
        self.nifti_affine_CT2S = None
        
        if self.data_type == 'atlas':
            self.obs_data['MWF_ATLAS'] = self.get_atlas_data()
            return
        
        if self.config.source.signal.T2 == True:
            
            ### for some reason, calling nii data by hlp-function is slower than using the
            ### commands directly in here --> overhead? incompatible format for memory?
            
            # img_nifti  = hlp.load_data(self.data_dir, self.config.source.file.T2)[0]
            # raw_data   = img_nifti.get_fdata()
            
            ### so we use this:
            img_nifti = nib.load(os.path.join(self.data_dir, self.config.source.file.T2)) 
            data_t2   = img_nifti.get_fdata()
            self.nifti_affine_T2 = img_nifti.affine

            if self.norm_max == True:
                max_val = np.max(data_t2, axis=3, keepdims=True)
                max_val[max_val == 0] = 1
                self.obs_data['T2'] = data_t2 / max_val
            else:
                self.obs_data['T2'] = data_t2
        
        if self.config.source.signal.T2S == True:
            
            img_nifti = nib.load(os.path.join(self.data_dir, self.config.source.file.T2S)) 
            data_t2s  = img_nifti.get_fdata()
            self.nifti_affine_T2S = img_nifti.affine
            
            if self.norm_max == True:
                max_val = np.max(data_t2s, axis=3, keepdims=True)
                max_val[max_val == 0] = 1                
                self.obs_data['T2S'] = data_t2s / max_val            
            else:
                self.obs_data['T2S'] = data_t2s  
                    
        if self.config.source.signal.CT2S==True:            
            
            self.obs_data['TE'] = np.load(os.path.join(self.data_dir, self.config.source.file.TE))
            
            img_nifti = nib.load(os.path.join(self.data_dir, self.config.source.file.T2S)) 
            data_T2S  = img_nifti.get_fdata()
            self.nifti_affine_T2S = img_nifti.affine
            
            img_nifti = nib.load(os.path.join(self.data_dir, self.config.source.file.CT2S)) 
            data_CT2S = img_nifti.get_fdata()
            filt_data = np.empty(data_CT2S.shape, dtype=np.complex128)
            self.nifti_affine_CT2S = img_nifti.affine
            
            for i in range(0,24):
                factor = 1.0                             # phase values in radiants
                if np.max(data_CT2S[:,:,i])>1000.0:      # assume unprocessed phase from dicom => -4096 < phase < 4096
                    factor             = np.pi/4096.0
                    filt_data[:,:,i,:] = data_T2S[:,:,i]*np.exp(1j*data_CT2S[:,:,i]*factor)
                
            if self.norm_max == True:
                max_val = np.max(np.abs(filt_data), axis=3, keepdims=True)
                max_val[max_val == 0] = 1
                self.obs_data['T2S' ] = filt_data / max_val
            else:
                self.obs_data['T2S' ] = filt_data 

        if self.config.source.signal.T1 == True:

            img_nifti = nib.load(os.path.join(self.data_dir, self.config.source.file.T1)) 
            data_t1   = img_nifti.get_fdata()
            self.nifti_affine_T1 = img_nifti.affine
                
            if self.norm_max == True:
                max_val = np.max(data_t1, axis=3, keepdims=True)
                max_val[max_val == 0] = 1
                self.obs_data['T1'] = data_t1 / max_val
            else:
                self.obs_data['T1'] = data_t1
            
    def calculate_sys_param(self, slice_calc: list):
        
        """
        Generates dictionary of system parameters for the slice(s) of interest
        used in the PSO application.
        """

        self.sys_param = {}        
        cpx            = self.config.source.signal.CT2S
        
        if self.data_type == 'invivo':
            
            # (a) execute class methods: invivo measurement data
            if self.verbose: 
                print('\nCalling measurement data ...')
                
            start_time    = time.time()
            
            self.get_obs_data(_data_type = self.data_type)
            
            if self.verbose:
                self.log(start_time=start_time, string='Execution time', dim='HMS')
            
            # (b) execute class methods: binary mask
            if self.verbose: 
                print(f'\nApplying mask status: {self.mask_use} ...')
                
            start_time    = time.time()
            self.handle_mask()
            
            if self.verbose:
                self.log(start_time=start_time, string='Execution time', dim='HMS')
            
            # (c) execute class methods: fill the system parameters dictionary 
            if self.verbose:
                print(f'\nCalculating system parameters for slice list - {slice_calc[:-1]} ...')
            
            start_time     = time.time()

            for i in range(slice_calc[0], slice_calc[-1]):

                self.access_mwf_analysis()           
                self.calc_system_cube(slice_num=i)
                    
                self.sys_param[f'{i:02}'] = {**{f'{sig}_MATRIX':  self.root_mwf.sm[sig]   for sig in self.decay_types},
                                             **{f'{sig}_MASK':    self.masks[sig][:,:,i] for sig in self.decay_types},
                                             **{f'{sig}_B1_GRID': self.b1_grid[sig]      for sig in self.decay_types},
                                             'B1_DATA':           self.root_mwf.data.slice['B1'],
                                             'CT2S_TE':           self.root_mwf.tsig['T2S'],
                                             'MeasData':          {sig: self.obs_data[sig][:,:,i] for sig in self.decay_types}}
                
            if self.verbose:
                self.log(start_time=start_time, string='Execution time', dim='HMS') 
            
        if self.data_type == 'atlas':
            
            # (a) execute class methods: invivo measurement data
            if self.verbose:
                print('\nCalling measurement data ...')
                
            start_time    = time.time()
            
            self.get_obs_data(_data_type = self.data_type)
            
            if self.verbose:
                self.log(start_time=start_time, string='Execution time', dim='HMS')
            
            # (b) get parametrization for generation of synthetic data
            data_path  = os.path.join(self.data_dir, self.config.source.file.ATLAS)
            snr_use    = self.config.source.data.add_noise[0]
            snr        = self.config.source.data.add_noise[1] if snr_use == True else 0
            config     = self.config.source.data.atlas
            axis       = config.axis
            seed       = config.seed
            mwf_thresh = config.thresh
            means      = [config.T1.mean, config.T2.mean, config.T2S.mean]
            widths     = [config.T1.width, config.T2.width, config.T2S.width]
            phases     = config.T2S.phase
            
            for i in range(slice_calc[0], slice_calc[-1]):
                
                self.masks = {'T1':[],'T2':[],'T2S':[]}
                
                self.access_mwf_analysis()
                self.calc_system_cube(slice_num=i)
                    
                if self.inv_T1 == True:                
                    self.masks['T1']  = self.root_mwf.prep_synthetic_data(data_path = data_path,
                                             axis = axis, slice_num = i, signal_type = 'T1', mwf_thresh = mwf_thresh,
                                             SNR  = snr, seed = seed,  dmean = np.array(means[0]), dstdv=np.array(widths[0]),
                                             x    = 120, y = 115, complex_T2S = False, phases=phases, verbose=False)

                if self.inv_T2 == True:
                    self.masks['T2']  = self.root_mwf.prep_synthetic_data(data_path = data_path,
                                             axis = axis, slice_num = i, signal_type = 'T2', mwf_thresh = mwf_thresh,
                                             SNR  = snr, seed = seed,  dmean = np.array(means[1]), dstdv=np.array(widths[1]),
                                             x    = 120, y = 115, complex_T2S = False, phases=phases, verbose=False)
    
                if self.inv_T2S == True:
                    self.masks['T2S'] = self.root_mwf.prep_synthetic_data(data_path = data_path,
                                             axis = axis, slice_num = i, signal_type = 'T2S', mwf_thresh = mwf_thresh,
                                             SNR  = snr, seed = seed,  dmean = np.array(means[2]), dstdv=np.array(widths[2]), 
                                             x    = 120, y = 115, complex_T2S = cpx, phases=phases, verbose=False)
                    
                    if cpx == True:
                        self.obs_data['T2S'] = self.root_mwf.data.slice['CT2S'].astype(np.complex128, copy=False)                        
                        self.obs_data['TE']  = self.root_mwf.tsig['T2S']                    
                
                # normalize measurement data to max value
                for sig in self.decay_types:                
                    
                    if not self.config.source.data.norm_max and not cpx:
                        self.obs_data[sig] = self.root_mwf.data.slice[sig]
                        continue
                
                    if cpx and sig == 'T2S':
                        num  = self.obs_data['T2S']
                        base = np.abs(self.root_mwf.data.slice['T2S'])
                    else:
                        num  = self.root_mwf.data.slice[sig]
                        base = num
                
                    max_val = np.max(base, axis=2, keepdims=True)
                    max_val[max_val == 0] = 1
                    self.obs_data[sig] = num / max_val                         
                    
                self.sys_param[f'{i:02}'] = {**{f'{sig}_MATRIX':  self.root_mwf.sm[sig]   for sig in self.decay_types},
                                             **{f'{sig}_MASK':    self.masks[sig][:,:,0] for sig in self.decay_types},
                                             **{f'{sig}_B1_GRID': self.root_mwf.b1_grid   for sig in self.decay_types},
                                             'B1_DATA':           self.root_mwf.data.slice['B1'],
                                             'CT2S_TE':           self.root_mwf.tsig['T2S'],
                                             'MeasData':          {sig: self.obs_data[sig] for sig in self.decay_types}}
        
        # execute class methods: preparing PSO constants
        if self.verbose:
            print('\nCalculating PSO system constants ...')       
        start_time      = time.time()
        self.constants = self._constant_PSO_objects(CT2S_TE=self.root_mwf.tsig['T2S'])
        
        if self.verbose:
            self.log(start_time=start_time, string='Execution time', dim='HMS')