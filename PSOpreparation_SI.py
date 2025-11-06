# -*- coding: utf-8 -*-
"""
Side script for preparing particle swarm optimization (PSO) on MRI invivo data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 08.2025; part of the JIMM/JIMM2 Project (DZNE Bonn & UFZ Leipzig)
"""

import os, time, copy
import numpy   as     np
import nibabel as     nib
from   pathlib import Path

import helpTools   as     hlp
from   mwf_t1t2t2s import mwf_analysis
from   PSOworkflow import ParticleSwarmOptimizer as PSOclass


###############################################################################
# global and constant parameter calculation ###################################
###############################################################################

class PSOpreparation(PSOclass):
    
    '''
    Prepatation of the PSO algorithm execution. Essential constants and parameters are 
    pre-defined; as well as measurement data are pre-processed here.
    
    For now, all functions are compatible to a single-inversion approach.
    
    Args:
        config_path: absolute path to your PSO configuration file
        mask_use:    ['config', 'calc', None] - config: mask is existing, calc: mask recently calculated
        mask_path:   if mask is recently calculated and stored to an absolute mask_path
        data_type:   ['meas', 'atlas']
    '''
    
    def __init__(self, config_input: str, config_type: str, mask_use=None, mask_path='', data_type='invivo'):
        
        # get configuration data and store into an object
        if config_type == 'file':
            self.config_path = config_input
            self.config2object()
        else:
            self.config      = config_input
        
        # some class objects
        self.data_type = data_type
        self.proj_dir  = self.config['source']['file']['prj_dir']
        self.data_dir  = os.path.normpath(os.path.join(self.proj_dir, 'nifti'))
        
        # about masking the data slice
        self.mask_use  = mask_use
        self.mask_path = mask_path
        
        # inherit PSO class methods
        super().__init__(self.config)
    
    def config2object(self):
        
        self.config = hlp.open_parameters(path=self.config_path)    
    
    def handle_mask(self):
        
        if self.mask_use == False:
            self.masks     = np.ones_like(self.obs_data.sig_list[0],dtype=int)
            
        if self.mask_use == True:
            self.mask_path = os.path.join(self.proj_dir, self.config.source.mask.path)
            mask_suffix    = Path(self.mask_path).suffixes[0]
            
            if mask_suffix == '.nii':
                mask_dummy     = nib.load(self.mask_path)
                self.masks     = mask_dummy.get_fdata()
                self.masks_raw = copy.deepcopy(self.masks)
                self.mask_roi  = self.config.source.mask.seg
                self.mask_roi = [x - 1 for x in self.mask_roi]
                
                self.masks = ((self.masks >= np.min(self.mask_roi)) &
                              (self.masks <= np.max(self.mask_roi))).astype(int)
                
            if mask_suffix == '.npy':
                self.masks     = np.load(file=self.mask_path,allow_pickle=True)
                
    
    def access_mwf_analysis(self):
        
        if self.data_type == 'atlas':
            self.rootMWF = mwf_analysis(data_dir = self.data_dir, 
                                        T2S_TE   = self.config.source.file.TE)
        
        if self.data_type == 'invivo':
            self.rootMWF = mwf_analysis(data_dir = self.data_dir,
                                        KW_B1    = self.config.source.file.B1,
                                        KW_T1    = self.config.source.file.T1,
                                        KW_T2    = self.config.source.file.T2,
                                        KW_T2S   = self.config.source.file.T2S,
                                        KW_T2SP  = self.config.source.file.CT2S,
                                        T2S_TE   = self.config.source.file.TE)


    def calc_system_cube(self, slice_num):
        
        if self.data_type == 'atlas':
            # no preparation function applied --> inhomogenity of B1 field ignored
            pass
        
        if self.data_type == 'invivo':
            
            for sig in self.signType:
                if self.config.source.signal.CT2S == True: sig = 'CT2S'
                
                self.rootMWF.prep_data(axis        = 'Z',
                                       slice_num   = slice_num,
                                       signal_type = sig,
                                       filter      = 1.0,
                                       thresh      = 4.5,
                                       verbose     = False)

        if self.invT2 == True:
            
            if self.data_type == 'invivo':
                self.rootMWF.prep_t2_model(te      = self.Inversion.T2_TE,         # 6.6  | sego: 6.0
                                           tr      = self.Inversion.T2_TR,         # 900  | sego: 2000
                                           etl     = self.Inversion.datSpaceT2,    # 24   | sego: 24
                                           alpha   = self.Inversion.T2_alpha,      # 70   | sego: 90
                                           beta    = self.Inversion.T2_beta,       # 180  | sego: 160
                                           T2min   = self.Inversion.T2min,         # 1    | sego: 1
                                           T2max   = self.Inversion.T2max,         # 200  | sego  200
                                           nT2     = self.Inversion.modSpace,      # 1000 | sego: 1000
                                           T1      = self.Inversion.T2_T1,         # 1000 | sego: 1000
                                           verbose = False)
            
            else:
                self.rootMWF.prep_t2_model(te      = 6.0,                          # 6.6  | sego: 6.0
                                           tr      = 2000,                         # 900  | sego: 2000
                                           etl     = self.Inversion.datSpaceT2,    # 24   | sego: 24
                                           alpha   = 90,                           # 70   | sego: 90
                                           beta    = 160,                          # 180  | sego: 160
                                           T2min   = self.Inversion.T2min,         # 1    | sego: 1
                                           T2max   = self.Inversion.T2max,         # 200  | sego  200
                                           nT2     = self.Inversion.modSpace,      # 1000 | sego: 1000
                                           T1      = self.Inversion.T2_T1,         # 1000 | sego: 1000
                                           verbose = False)
            
            
        if self.invT2S == True:
            self.rootMWF.prep_t2s_model(T2Smin  = self.Inversion.T2Smin,
                                        T2Smax  = self.Inversion.T2Smax,
                                        nT2S    = self.Inversion.modSpace,
                                        verbose = False)
    
        if self.invT1 == True:                
            self.rootMWF.prep_t1_model(tr      = self.Inversion.T1_TR,
                                       alpha   = self.Inversion.T1_alpha,
                                       td      = self.Inversion.T1_TD,
                                       ie      = self.Inversion.T1_IE,
                                       T1min   = self.Inversion.T1min,
                                       T1max   = self.Inversion.T1max,
                                       nT1     = self.Inversion.modSpace,
                                       verbose = False)
        
    def get_atlas_data(self):        
                        
        mwf_mean  = nib.load(os.path.join(self.data_dir, self.config.source.file.ATLAS))
        mwf_data  = mwf_mean.get_fdata()
        
        return mwf_data
    
    def get_obs_data(self, _filter_bool: bool(), _filter_vals: tuple(), _signal=None, _data_type='invivo'):
        
        self.obs_data = {'T1':[],'T2':[],'T2S':[],'CT2S':[],'TE':[],'MWF_ATLAS':[]}
        self.norm_max = self.config.source.data.norm_max
        
        if _signal:
            for sig in ("T1", "T2", "T2S", "CT2S"):
                setattr(self.config.source.signal, sig, sig==_signal)
        
        print(f'Spatial gaussian smoothing filter: {_filter_bool}')
        
        if self.data_type == 'atlas':
            self.obs_data['MWF_ATLAS'] = self.get_atlas_data()
            return
        
        if self.config.source.signal.T2 == True:
            
            ### for some reason, calling nii data by hlp-function is slower than using the
            ### commands directly in here --> overhead? incompatible format for memory?
            
            # img_nifti  = hlp.load_data(self.data_dir, self.config.source.file.T2)[0]
            # raw_data   = img_nifti.get_fdata()
            
            ### so we use this:
            img_nifti   = nib.load(os.path.join(self.data_dir, self.config.source.file.T2)) 
            raw_data_T2 = img_nifti.get_fdata()
            filt_data   = np.empty_like(raw_data_T2)
            
            if _filter_bool == True:
                from scipy.ndimage import gaussian_filter
                for i in range(0,24):
                    filt_data[:,:,i] = gaussian_filter(raw_data_T2[:,:,i], _filter_vals)
                
                if self.norm_max == True:
                    max_val = np.max(filt_data, axis=3, keepdims=True)
                    max_val[max_val == 0] = 1
                    self.obs_data['T2'] = filt_data / max_val
                else:
                    self.obs_data['T2'] = filt_data 
            
            if _filter_bool == False:                
                if self.norm_max == True:
                    max_val = np.max(raw_data_T2, axis=3, keepdims=True)
                    max_val[max_val == 0] = 1
                    self.obs_data['T2'] = raw_data_T2 / max_val
                else:
                    self.obs_data['T2'] = raw_data_T2
        
        if self.config.source.signal.T2S == True:

            img_nifti    = nib.load(os.path.join(self.data_dir, self.config.source.file.T2S)) 
            raw_data_T2S = img_nifti.get_fdata()
            filt_data    = np.empty_like(raw_data_T2S)
            
            if _filter_bool == True:
                from scipy.ndimage import gaussian_filter                
                for i in range(0,24):
                    filt_data[:,:,i] = gaussian_filter(raw_data_T2S[:,:,i], _filter_vals)
                
                max_val = np.max(filt_data, axis=3, keepdims=True)
                max_val[max_val == 0] = 1
                
                self.obs_data['T2S'] = filt_data / max_val
            
            if _filter_bool == False:
                
                max_val = np.max(raw_data_T2S, axis=3, keepdims=True)
                max_val[max_val == 0] = 1
                
                self.obs_data['T2S'] = raw_data_T2S / max_val        
                    
        if self.config.source.signal.CT2S==True:
            
            self.obs_data['TE'] = np.load(os.path.join(self.data_dir, self.config.source.file.TE))
            
            img_nifti     = nib.load(os.path.join(self.data_dir, self.config.source.file.T2S)) 
            raw_data_T2S  = img_nifti.get_fdata()
            
            img_nifti     = nib.load(os.path.join(self.data_dir, self.config.source.file.CT2S)) 
            raw_data_CT2S = img_nifti.get_fdata()
            filt_data     = np.empty(raw_data_CT2S.shape, dtype=np.complex128)
            
            for i in range(0,24):
                factor = 1.0                                 # phase values in radiants
                if np.max(raw_data_CT2S[:,:,i])>1000.0:      # assume unprocessed phase from dicom => -4096 < phase < 4096
                    factor             = np.pi/4096.0
                    filt_data[:,:,i,:] = raw_data_T2S[:,:,i]*np.exp(1j*raw_data_CT2S[:,:,i]*factor)
                
                if _filter_bool == True:
                    from scipy.ndimage import gaussian_filter
                        
                    filt_real = gaussian_filter(np.real(filt_data[:,:,i]),_filter_vals)  # np.angle takes the phase
                    filt_imag = gaussian_filter(np.imag(filt_data[:,:,i]),_filter_vals)  # np.abs takes the magnitude
                    
                    filt_data[:,:,i,:] = (filt_real + 1j*filt_imag)
                    
            max_val = np.max(np.abs(filt_data), axis=3, keepdims=True)
            max_val[max_val == 0] = 1
            self.obs_data['T2S' ] = filt_data / max_val


    def calculate_sys_param(self, slice_calc: list):

        self.sys_param = {}
        sig            = self.signType[0]
        cpx            = self.config.source.signal.CT2S
        
        if self.data_type == 'invivo':
            
            # (a) execute class methods: invivo measurement data
            print('\nCalling measurement data ...')
            startTime     = time.time()
            
            self.get_obs_data(_filter_bool = self.config.source.data.gauss_filt[0],
                              _filter_vals = tuple(self.config.source.data.gauss_filt[1]),
                              _data_type   = self.data_type)
            
                    
            self.log(startTime=startTime, string='Execution time', dim='HMS')
            
            # (b) call mask for the system
            # can be later integrated by here
            print(f'\nApplying mask status: {self.mask_use} ...')
            startTime     = time.time()
            self.handle_mask()
            self.log(startTime=startTime, string='Execution time', dim='HMS')
            
            # (c) execute class methods: preparing system parameters - grid, mask, matrix, ... 
            print(f'\nCalculating system parameters for slice list - {slice_calc[:-1]} ...')
            startTime      = time.time()

            for i in range(slice_calc[0], slice_calc[-1]):

                self.access_mwf_analysis()           
                self.calc_system_cube(slice_num=i)
                
                self.sys_param[f'{i:02}'] = {f'{sig}_MATRIX': self.rootMWF.sm['T2'],
                                             'B1_DATA':       self.rootMWF.data.slice['B1'],
                                             'T2S_MATRIX':    self.rootMWF.sm['T2S'],
                                             'B1_GRID':       self.rootMWF.b1_grid,
                                             f'{sig}_MASK':   self.masks[:,:,i],
                                             'CT2S_TE':       self.rootMWF.tsig['T2S'],
                                             'MeasData':      {self.signType[0]: self.obs_data[self.signType[0]][:,:,i]}}
            
        if self.data_type == 'atlas':
            
            # (a) execute class methods: invivo measurement data
            print('\nCalling measurement data ...')
            startTime     = time.time()
            
            self.get_obs_data(_filter_bool = self.config.source.data.gauss_filt[0],
                              _filter_vals = tuple(self.config.source.data.gauss_filt[1]),
                              _data_type   = self.data_type)
            

            self.log(startTime=startTime, string='Execution time', dim='HMS')
            
            for i in range(slice_calc[0], slice_calc[-1]):

                SNR_use    = self.config.source.data.add_noise[0]
                SNR        = self.config.source.data.add_noise[1] if SNR_use == True else 0
                self.masks = {'T1':[],'T2':[],'T2S':[]}
                
                self.access_mwf_analysis()
                self.calc_system_cube(slice_num=i)
                
                if sig == 'T1':
                    pass
                
                if sig == 'T2':
                    self.masks['T2']  = self.rootMWF.prep_synthetic_data(
                                             axis = 'Z', slice_num = i, signal_type = 'T2', 
                                             SNR  = SNR, seed = 1,  dmean = np.array([20,70]), dstdv=np.array([0.5,0.5]),
                                             x    = 120, y = 115, complex_T2S = False, phases=[-8,0,0.5], verbose=False)
                
                if sig == 'T2S':
                    self.masks['T2S'] = self.rootMWF.prep_synthetic_data(
                                             axis = 'Z', slice_num = i, signal_type = 'T2S',
                                             SNR  = SNR, seed = 1,  dmean = np.array([15,60]), dstdv=np.array([0.5,0.5]), 
                                             x    = 120, y = 115, complex_T2S = cpx, phases=[-8,0,0.5], verbose=False)
                    
                    if cpx == True:
                        # complex array contains only the phase !!! which is not being normalized
                        # self.obs_data['CT2S'] = np.unwrap(np.angle(self.rootMWF.data.slice['CT2S']))
                        
                        self.obs_data['T2S'] = self.rootMWF.data.slice['CT2S'].astype(np.complex128, copy=False)                        
                        self.obs_data['TE']  = self.rootMWF.tsig['T2S']
                    
                # normalize measurement data
                if self.config.source.data.norm_max == True and cpx == False:
                    max_val = np.max(self.rootMWF.data.slice[sig], axis=2, keepdims=True)
                    max_val[max_val == 0] = 1
                    self.obs_data[sig]    = self.rootMWF.data.slice[sig]/max_val
                
                elif self.config.source.data.norm_max == True and cpx == True:
                    max_val = np.max(np.abs(self.rootMWF.data.slice['T2S']), axis=2, keepdims=True)
                    max_val[max_val == 0] = 1
                    self.obs_data['T2S' ] = self.obs_data['T2S' ]/max_val
    
                elif self.config.source.data.norm_max == False and cpx == False:
                    self.obs_data[sig] = self.rootMWF.data.slice[sig]
                
                self.sys_param[f'{i:02}'] = {f'{sig}_MATRIX': self.rootMWF.sm[sig],
                                             'B1_DATA':       self.rootMWF.data.slice['B1'],
                                             'B1_GRID':       self.rootMWF.b1_grid,
                                             f'{sig}_MASK':   self.masks[sig][:,:,0],
                                             'CT2S_TE':       self.rootMWF.tsig['T2S'],
                                             'MeasData':      {sig: self.obs_data[sig]}}
        
        # execute class methods: preparing PSO constants
        print('\nCalculating PSO system constants ...')       
        startTime      = time.time()
        self.constants = self._constant_PSO_objects_test(CT2S_TE=self.rootMWF.tsig['T2S'])
        
        self.log(startTime=startTime, string='Execution time', dim='HMS')