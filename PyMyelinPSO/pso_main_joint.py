# SPDX-FileCopyrightText: 2026 Helmholtz-Zentrum f�r Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Author:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of PyMyelinPSO.
# See the LICENSE file in the project root for license information.

"""
Main script for performing particle swarm optimization (PSO) on MRI invivo data.
All features are controlled via a user-defined configuration dictionary.
Inversion type: joint inversion.

Contains:
    1) Preparation of the PSO environment based on the configuration
    2) Setup of parallel execution on multi-core systems
    3) Execution of PSO in the selected mode, including saving/returning results
"""

import os

###############################################################################
# Environment configuration (set before importing NumPy / SciPy)
# --------------------------------------------------------------
# Restricts internal thread usage of math libraries (e.g. MKL, OpenBLAS)
# to prevent thread oversubscription — especially important on Linux systems.
###############################################################################
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

###############################################################################
# Import of required libraries (built-in and third-party)
###############################################################################
import os, time, concurrent.futures
import numpy    as     np
import nibabel  as     nib
from   scipy.io import savemat
from   pathlib  import Path

###############################################################################
# Import of project-specific modules
# --------------------------------------------------------------
# Custom tools and classes must reside in the same directory as the main script.
###############################################################################
import help_tools            as     hlp
from   pso_core              import ParticleSwarmOptimizer as PSOclass
from   pso_preparation_joint import PSOpreparation_JI      as PSOprep
from   pso_visualization     import PSOPlotter

###############################################################################
# Worker initialization and memory mapping utilities
# --------------------------------------------------
# These functions handle shared-memory setup between the main process and
# worker processes. They create and open memory-mapped arrays, broadcast
# configuration data, and ensure consistent initialization across all workers.
###############################################################################

# Global dictionary for data shared between worker processes
_G = {}

# Basic worker initializer (runs once per worker when the pool starts)
def _init_worker_base():
    _G.clear()                                       # reset any previous state
    _G['configured'] = False  # flag: worker not yet configured with slice data

# Create a memory-mapped file from a NumPy array (main process only)
def _mk_memmap_from_array(arr: np.ndarray, base_dir: str, name: str):
    
    '''
    Writes a NumPy array as a C-contiguous memmap and returns metadata
    (path, shape, dtype, order) for worker access.
    '''
    
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path     = str(base_dir / f"{name}.dat")

    a        = np.ascontiguousarray(arr)                       # ensure C-order
    mm       = np.memmap(path, dtype=a.dtype, mode="w+", shape=a.shape)
    mm[:]    = a[:]                                       # write array to disk
    
    del mm                                                      # flush & close
    return path, a.shape, a.dtype.str, "C"

# Open an existing memmap read-only in a worker
def _open_memmap_ro(path: str, shape, dtype_str, order="C"):
    
    '''
    Reopens a memmap file in read-only mode for workers.
    '''
    
    return np.memmap(path, dtype=np.dtype(dtype_str), mode="r", shape=tuple(shape), order=order)

# Configure worker for current slice by loading all shared memmaps
def _switch_slice(memdesc: dict, b1_index_map, sig_list, config_static, constants):

    '''
    Loads memmaps and config into worker’s global memory.
    '''
    sig1, sig2       = sig_list[0], sig_list[1]
    
    _G['config']     = config_static
    _G['constants']  = constants
    _G['sig_list']   = sig_list
    _G['b1_idx_map'] = b1_index_map

    _G[f'{sig1}_MATRIX']   = _open_memmap_ro(*memdesc[f'{sig1}_MATRIX'])
    _G[f'{sig2}_MATRIX']   = _open_memmap_ro(*memdesc[f'{sig2}_MATRIX'])
    _G[f'{sig1}_MASK']     = _open_memmap_ro(*memdesc[f'{sig1}_MASK'])
    _G[f'{sig1}_B1_GRID']  = _open_memmap_ro(*memdesc[f'{sig1}_B1_GRID'])
    _G[f'{sig2}_B1_GRID']  = _open_memmap_ro(*memdesc[f'{sig2}_B1_GRID'])
    _G[f'{sig1}_MEAS_SIG'] = _open_memmap_ro(*memdesc[f'{sig1}_MEAS_SIG'])
    _G[f'{sig2}_MEAS_SIG'] = _open_memmap_ro(*memdesc[f'{sig2}_MEAS_SIG'])
    _G['CT2S_TE']          = _open_memmap_ro(*memdesc['CT2S_TE'])
    _G['B1_DATA']          = _open_memmap_ro(*memdesc['B1_DATA'])
    
    _G['configured']       = True

    return os.getpid()

# Broadcasts slice initialization to all workers (ensures all are configured)
def _broadcast_switch(executor, memdesc, b1_idx_map, sig_list, config, constants, *, max_workers=None):
    
    '''
    Ensures all workers execute _switch_slice() once with current slice data.
    '''
    
    max_workers = max_workers or getattr(executor, "_max_workers", os.cpu_count() or 1)

    seen = set()

    while len(seen) < max_workers:
        need = max_workers - len(seen)
        futs = [executor.submit(_switch_slice, memdesc, b1_idx_map, sig_list, config, constants)
                for _ in range(need)]
        for f in concurrent.futures.as_completed(futs):
            seen.add(f.result())
            
###############################################################################
# Pixel-level PSO execution and parallelization utilities
# -------------------------------------------------------
# These functions manage per-pixel PSO runs using shared slice data (_G),
# build pixel parameter lists for multiprocessing, and execute tasks in parallel.
###############################################################################

# Executes the PSO algorithm for a single pixel using shared slice data
def _run_pso_pixel(position):

    '''
    Runs the PSO algorithm for one pixel using shared slice data.
    Builds pixel-specific parameters, executes optimization, and returns results.
    '''
    
    if not _G.get('configured', False):
        raise RuntimeError("Worker not initialized. Run _broadcast_switch before pixel tasks.")

    ##### unpack pixel coordinates and random seed
    yy, xx, rand = position

    ##### access shared global data (static per slice)
    cfg_base   = _G['config']
    constants  = _G['constants']
    sig_list   = _G['sig_list']
    b1_idx_map = _G['b1_idx_map']
    sig1, sig2, sigJI = sig_list[0], sig_list[1], sig_list[-1]
    
    ##### construct local system parameters for this pixel
    sys_param_ = {**{f'{sig}_MATRIX': _G[f'{sig}_MATRIX'][:, :, b1_idx_map[sig][yy, xx]] for sig in sig_list[:2]},
                     f'{sig1}_MASK':  _G[f'{sig1}_MASK'],
                      'OBS_DATA':     {sig: _G[f'{sig}_MEAS_SIG'][yy, xx, :] for sig in sig_list[:2]},
                      'CT2S_TE':      _G['CT2S_TE'],}
    
    ##### adjust dynamic, mathematic parameters in config using flat copies
    config               = dict(cfg_base)
    PSO_spec             = dict(config['PSO_spec'])
    PSO_math             = dict(PSO_spec['PSO_math'])
    PSO_math['pixel']    = [yy, xx]
    PSO_math['rand']     = list(PSO_math['rand']); PSO_math['rand'][1] = rand
    PSO_spec['PSO_math'] = PSO_math
    config['PSO_spec']   = PSO_spec

    ##### initialize and execute PSO for this pixel
    PSO = PSOclass(config_data=config, sys_param=sys_param_, 
                   position=(yy,xx), init_matrix=True, 
                   constants=constants)

    try:
        PSO.run_pso(obs_decay=sys_param_['OBS_DATA'], plot_iter_test=False)

        return {'pix': [yy,xx],
                f'mod{sig1}':      PSO.glob_mod[sig1], 
                f'mod{sig2}':      PSO.glob_mod[sig2],
                'fit':             PSO.glob_fit[sigJI],    
                f'syn_data{sig1}': PSO.glob_syn_data[sig1], 
                f'syn_data{sig2}': PSO.glob_syn_data[sig2]}
    
    finally:
        PSO._close()
        del PSO

# Create pixel parameter lists for parallelized PSO execution
def _set_pix_param(sig_list: list, rand: int, slice_param: dict, data_type='invivo'):
    
    '''
    Builds a list of pixel coordinates (y, x, rand) 
    and a B1 index map used for parallel PSO execution on a slice.    
    '''
    
    B1_GRID = {item: slice_param[f'{item}_B1_GRID'] for item in sig_list}
    B1_DATA = slice_param['B1_DATA']

    # we only take one mask for all signals
    mask      = slice_param[f'{sig_list[0]}_MASK']  
    coords    = np.argwhere(mask == 1)
    
    positions = [(int(y), int(x), int(rand)) for y, x in coords]
    
    if data_type == 'atlas':
        return positions, {sig: np.zeros_like(B1_DATA, dtype=np.int16) for sig in sig_list}

    if data_type == 'invivo':
        b1_idx_map = {sig: np.empty_like(B1_DATA, dtype=np.int16) for sig in sig_list}
        
        for sig in sig_list:            
            for (yy, xx), _ in np.ndenumerate(B1_DATA):
                b1_idx_map[sig][yy, xx] = np.argmin(np.abs(B1_GRID[sig] - B1_DATA[yy, xx]))
    
        return positions, b1_idx_map

# Parallel execution of PSO over all pixels in the given list
def main_parallel_pixels(executor: object, pix_list: list, chunksize=1):
    return list(executor.map(_run_pso_pixel, pix_list, chunksize=chunksize))

###############################################################################
# PSO execution control
# --------------------------------------------------
# Controls the complete PSO workflow: loads configuration, validates parameters, 
# prepares system data via PSOpreparation, and initializes slice- or pixel-based runs.
###############################################################################

def run_pso_on_config(config_input: str, config_type: str, save_results=True):
    
    '''
    Initialization of the PSO environment and preparation of system parameters based on user input.

    Args:
        config_input: dictionary or full configuration file path
        config_type:  ['dict', 'file']
        save_results: only used for atlas-based inversion, target: full cube
    '''

    # Console log
    print('Running PSO preparation (e.g. obs_data, system matrices, system constants) ...') 
    start_time_overall = time.time()
    
    # Load configuration (from file or dict)
    if config_type == 'file':
        config_path = config_input
        config      = hlp.open_parameters(path=config_path)
    if config_type == 'dict':
        config      = config_input

    # Validate configuration
    config      = hlp.troubleshooting(config=config)

    # Extensive console printout
    verbose     = config['general']['verbose']

    # Initialize main PSO class and plotting utilities
    _PSO        = PSOclass(config_data=config)
    pso_plotter = PSOPlotter()  
    
    # Define signal list and primary signal type# defining variables
    _sig_list  = [k for k in list(config['source']['signal'].keys())[:3] if config['source']['signal'][k]]
    _signal_JI = '_'.join(_sig_list)
    
    if config['source']['signal']['CT2S'] and config['source']['signal']['T2S']: 
        _signal_JI = _signal_JI.replace('T2S', 'CT2S')
        
    _sig_list.append(_signal_JI)

    # Determine slices for processing
    calc = config['PSO_spec']['comp_mode']
    if calc['iter_test']['use']:
        _slice_calc = [calc['iter_test']['slice'], calc['iter_test']['slice']+1]
    elif calc['performance_test']['use']:
        _slice_calc = [7, 8]
    elif calc['PSO_on_pixel']['use']:
        s = calc['PSO_on_pixel']['slice']; _slice_calc = [s, s + 1]
    elif calc['PSO_on_slice']['use']:
        s0, s1 = calc['PSO_on_slice']['start'], calc['PSO_on_slice']['end']
        _slice_calc = list(range(s0, s1 + 1))

    # Prepare PSO analysis and system parameters
    pre_analysis = PSOprep(config_input = config_input,
                           config_type  = config_type,
                           mask_use     = config['source']['mask']['use'],
                           mask_path    = {sig: config['source']['mask']['path'] for sig in _sig_list[:2]},
                           data_type    = config['source']['data']['type'])
    
    pre_analysis.calculate_sys_param(slice_calc=_slice_calc)  
    sys_param    = pre_analysis.sys_param
    _signal_src  = pre_analysis.data_source
    
###############################################################################    
### (b.1) Parallelized calculation of MWF for one slice and with n PSO cycles

    if pre_analysis.config.PSO_spec.comp_mode.PSO_on_slice.use == True:

        if verbose:
            print('\nInitialize shared PSO resources for parallel workers (memmaps, shared data, ...)')         
            start_time_pre = time.time()

        PSO_spec      = pre_analysis.config.PSO_spec
        PSO_source    = pre_analysis.config.source
        
        if _signal_src=='invivo':
            run_sig1_file = getattr(PSO_source.file, _sig_list[0])
            run_sig2_file = getattr(PSO_source.file, _sig_list[1])
            run_sig1 = run_sig1_file.removesuffix('.gz').removesuffix('.nii')
            run_sig2 = run_sig2_file.removesuffix('.gz').removesuffix('.nii')

        if _signal_src == 'invivo':
            _run_sig1 = run_sig1.split('_')
            _run_sig2 = run_sig2.split('_')
            if _run_sig1[-2] == _run_sig2[-2] and _run_sig1[-1] == _run_sig2[-1]:
                ID_string = f'{_run_sig1[-2]}_{_run_sig1[-1]}'
            else:
                ID_string = f'{_sig_list[0]}_{_run_sig1[-2]}_{_run_sig1[-1]}_{_sig_list[1]}_{_run_sig2[-2]}_{_run_sig1[-1]}'
            
        elif _signal_src == 'atlas':
            ID_string = ''
        
        # path e.g.:
        #       *prj_dir/results/T2_T2S_JI/avg_degibbsed/015I032P100PSO/T2/2_comp/ 
        _save_dir = os.path.normpath(os.path.join(PSO_source.file.prj_dir, 
                                                  'results', f'{_signal_JI}_JI', ID_string,
                                                  f'{str(PSO_spec.PSO_math.n_iter).zfill(3)}I'
                                                  f'{str(PSO_spec.PSO_math.n_part).zfill(3)}P'
                                                  f'{str(PSO_spec.PSO_math.cyc_slice).zfill(2)}PSO',
                                                  f'{str(PSO_spec.PSO_math.n_comp)}_comp'))
        
        _PSO_iter = PSO_spec.PSO_math.cyc_slice
        _n_comp   = PSO_spec.PSO_math.n_comp
        _no_param = {sig: pre_analysis.n_param[sig]+1 for sig in _sig_list[:2]}
        _no_steps = {sig: pre_analysis.n_echoes[sig]  for sig in _sig_list[:2]}
        os.makedirs(_save_dir, exist_ok=True)
        
        # random necessary for comparability of PSO runs with different parameters
        np.random.seed(0)
        max_workers = 56
        slice_size, slice_count = PSO_source.data.shape[:2], PSO_source.data.shape[2]
    
        # Log executor time
        start_time_cyc = time.time()

        # execute parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_base) as executor:
            
            #print(slice_size[0], slice_size[1], _no_param[_sig_list[0]], _PSO_iter+1, slice_count)
            
            full_cube = {sig: np.zeros([slice_size[0], slice_size[1], _no_param[sig], _PSO_iter+1, slice_count],
                                       dtype=np.float32) for sig in _sig_list[:2]}
    
            #import sys; sys.exit()
            
            for i, item in enumerate(range(slice_count)[_slice_calc[0]:_slice_calc[-1]]):
                
                start_time_slc = time.time()
            
                # definition of essential and execution parameters for each slice
                key            = str(item).zfill(2)
                slice_param    = sys_param[key]
                
                rand_vector    = list()
                save_path      = f'{_save_dir}\\slice{str(item).zfill(2)}\\'               
                
                result_map_sl  = {sig: np.zeros([slice_size[0], slice_size[1], _no_param[sig], _PSO_iter+1], dtype=np.float32) for sig in _sig_list[:2]}
                syn_dat_map    = {sig: np.empty((slice_size[0], slice_size[1], _no_steps[sig], _PSO_iter), dtype=np.float32) for sig in _sig_list[:2]}
                res_array_dic  = {sig: np.empty((slice_size[0], slice_size[1], _PSO_iter), dtype=object) for sig in _sig_list[:2]}
                
                for sig in _sig_list[:2]:
                    mask0                           = slice_param[f'{sig}_MASK'] == 0
                    result_map_sl[sig][mask0, :, :] = np.nan
                    syn_dat_map[sig][mask0, :, :]   = np.nan
                                
                rand_seeds                          = np.random.randint(0, 100001, size=_PSO_iter)
                _, b1_idx_map                       = _set_pix_param(_sig_list[:2], 0, slice_param, pre_analysis.data_type)
            
                # create memmaps for this slice (in a slice-specific directory)
                tmpdir = Path(os.path.join(_save_dir, f'Slice{key}', '_mm')).resolve()
                tmpdir.mkdir(parents=True, exist_ok=True)
                
                # the first iteration pays the cost of making data visible; later iterations just reuse it
                # (writing mode "w+")
                sig1, sig2 = _sig_list[0], _sig_list[1]
                memdesc = {f'{sig1}_MATRIX':   _mk_memmap_from_array(slice_param[f'{sig1}_MATRIX'],  tmpdir, "f'{sig1}_MATRIX"),
                           f'{sig2}_MATRIX':   _mk_memmap_from_array(slice_param[f'{sig2}_MATRIX'],  tmpdir, "f'{sig2}_MATRIX"),
                           f'{sig1}_MASK':     _mk_memmap_from_array(slice_param[f'{sig1}_MASK'],    tmpdir, "f'{sig1}_MASK"),
                           f'{sig1}_B1_GRID':  _mk_memmap_from_array(slice_param[f'{sig1}_B1_GRID'], tmpdir, "f'{sig1}_B1_GRID"),
                           f'{sig2}_B1_GRID':  _mk_memmap_from_array(slice_param[f'{sig2}_B1_GRID'], tmpdir, "f'{sig2}_B1_GRID"),
                           f'{sig1}_MEAS_SIG': _mk_memmap_from_array(slice_param['MeasData'][sig1],  tmpdir, "f'{sig1}_MEAS_SIG"),
                           f'{sig2}_MEAS_SIG': _mk_memmap_from_array(slice_param['MeasData'][sig2],  tmpdir, "f'{sig2}_MEAS_SIG"),
                           'CT2S_TE':          _mk_memmap_from_array(slice_param['CT2S_TE'],    tmpdir, "CT2S_TE"),
                           'B1_DATA':          _mk_memmap_from_array(slice_param['B1_DATA'],    tmpdir, "B1_DATA"),}
                
                _broadcast_switch(executor    = executor,
                                  memdesc     = memdesc,
                                  b1_idx_map  = b1_idx_map,
                                  sig_list    = _sig_list,
                                  config      = config,
                                  constants   = pre_analysis.constants,
                                  max_workers = max_workers,)
                
                # time log for initialization of shared PSO resources for parallel workers
                if verbose and i == 0:
                    _PSO.log(start_time=start_time_pre, string='Execution time', dim='HMS')
                    print(f'\nStarting PSO joint inversion - signal: {_signal_JI} | components: {_n_comp}')

                # time log for preparation time of PSO algorithm
                if not verbose and i == 0:
                    _PSO.log(start_time=start_time_overall, string='Execution time', dim='HMS')
                    print(f'\nStarting PSO joint inversion - signal: {_signal_JI} | components: {_n_comp}')
                    
                # PSO routine iterated over different initial randomization
                for kk, rand in enumerate(rand_seeds):

                    rand_vector.append((f'{str(kk).zfill(2)}', rand))
                    
                    positions, _ = _set_pix_param(_sig_list[:2], rand, slice_param)              
                    results_sl   = main_parallel_pixels(executor=executor, pix_list=positions, chunksize=1)

                    result_map_sl, syn_dat_map, res_array_dic = _PSO.dict_to_array(
                                                                    results_sl, result_map_sl, syn_dat_map, res_array_dic, kk,
                                                                    calcBestResult = True,
                                                                    arrayType      = 'Slice')

                # plot function param_map from PSOPlotter class
                    # for plot_sig in pre_analysis.decay_types:
                    #     pso_plotter.param_map_multi(inv_data=result_map_sl[plot_sig], pso_class=pre_analysis, signal=plot_sig, idx=item,
                    #                                 save_path=f'{save_path}{plot_sig}_sl{key}_param.png', save_format='PNG', save_dpi=300, save=True)                    
                
                    if kk==_PSO_iter-1:
                        for plot_sig in pre_analysis.decay_types:
                            pso_plotter.param_map_multi(inv_data=result_map_sl[plot_sig], pso_class=pre_analysis, signal=plot_sig, idx=-1,
                                                        save_path=f'{save_path}{plot_sig}_bestfit_param.png', save_format='PNG', save_dpi=300, save=True)

                # fill full brain cube
                for sig in _sig_list[:2]:
                    full_cube[sig][..., item] = result_map_sl[sig]            
                
                # write and return results
                np.savetxt(f'{save_path}random_seeds.txt',
                           np.array(rand_vector, dtype=object), fmt="%s %s", delimiter=" ",
                           header="PSO_Iteration random_seed", comments="")
    
                for sig in _sig_list[:2]:                    
                    nib.save(nib.Nifti1Image(result_map_sl[sig], affine=np.eye(4)), 
                             f'{save_path}{sig}_results_modelvector.nii.gz')                
                    
                    nib.save(nib.Nifti1Image(syn_dat_map[sig],
                                             affine=getattr(pre_analysis, f"nifti_affine_{sig}")),
                                             f'{save_path}{sig}_results_syndata.nii.gz')
                    
                    
                    savemat(f'{save_path}{sig}_results_modelvector.mat', result_map_sl, do_compression=True)
                    savemat(f'{save_path}{sig}_results_syndata.mat',     syn_dat_map, do_compression=True)

                _PSO.write_json(path=f'{save_path}')
                
                # time log slice
                _PSO.log(start_time=start_time_slc, string=f'\nExecution time slice {str(item).zfill(2)}', dim='HMS')
       
        # saving takes time - no saving of full cube if atlas is inverted
        if not save_results:
            return result_map_sl[_sig_list[0]], result_map_sl[_sig_list[1]], pre_analysis
        
        # write full data cubes for each model vector parameter and time log
        if _n_comp == 2:
            MWF_joint = np.mean(np.stack([full_cube[_sig_list[0]][:,:,5], 
                                          full_cube[_sig_list[1]][:,:,5]], axis=0), axis=0)
            for sig in _sig_list[:2]:
                param_inst = getattr(_PSO, f'{sig}').TwoComponentParams
                param_keys = list(vars(param_inst).keys())
                if _PSO.inv_CT2S:
                    param_keys.append('misfit')
                else:
                    param_keys = param_keys[:6]; param_keys.append('misfit')                
            
        if _n_comp == 3:
            MWF_joint = np.mean(np.stack([full_cube[_sig_list[0]][:,:,9], 
                                          full_cube[_sig_list[1]][:,:,9]], axis=0), axis=0)
            for sig in _sig_list[:2]:
                param_inst = getattr(_PSO, f'{sig}').ThreeComponentParams
                param_keys = list(vars(param_inst).keys())
                if _PSO.inv_CT2S:
                    param_keys.append('misfit')
                else:
                    param_keys = param_keys[:9]; param_keys.append('misfit')
        
        for idx in range(full_cube[_sig_list[0]].shape[2]):
            nib.save(nib.Nifti1Image(MWF_joint.transpose(0,1,3,2), 
                                     affine=getattr(pre_analysis, f"nifti_affine_{_sig_list[0]}")),
                                     f'{_save_dir}\\mwf_joint_full_brain.nii.gz')             
            for sig in _sig_list[:2]:
                nib.save(nib.Nifti1Image(full_cube[sig][:,:,idx,:,:].transpose(0,1,3,2), 
                                         affine=getattr(pre_analysis, f"nifti_affine_{sig}")),
                                         f'{_save_dir}\\{sig}_{param_keys[idx].lower()}_full_brain.nii.gz')        

        _PSO.log(start_time=start_time_cyc, string=f'\nExecution time {_PSO_iter} PSO cycles', dim='HMS')        
        print(f'Saved results to {os.path.dirname(os.path.normpath(save_path))} ...')
        
        return result_map_sl[_sig_list[0]], result_map_sl[_sig_list[1]], pre_analysis
            
#########################################################################################################

### (b.2) Parallelized calculation of MWF for one pixel with different randoms

    if pre_analysis.config.PSO_spec.comp_mode.PSO_on_pixel.use==True:
        
        if verbose:
            print('\nInitialize shared PSO resources for parallel workers (memmaps, shared data, ...)')         
            start_time_pre = time.time()

        PSO_spec      = pre_analysis.config.PSO_spec
        PSO_source    = pre_analysis.config.source
        
        if _signal_src=='invivo':
            run_sig1_file = getattr(PSO_source.file, _sig_list[0])
            run_sig2_file = getattr(PSO_source.file, _sig_list[1])
            run_sig1 = run_sig1_file.removesuffix('.gz').removesuffix('.nii')
            run_sig2 = run_sig2_file.removesuffix('.gz').removesuffix('.nii')

        if _signal_src == 'invivo':
            _run_sig1 = run_sig1.split('_')
            _run_sig2 = run_sig2.split('_')
            if _run_sig1[-2] == _run_sig2[-2] and _run_sig1[-1] == _run_sig2[-1]:
                ID_string = f'{_run_sig1[-2]}_{_run_sig1[-1]}'
            else:
                ID_string = f'{_sig_list[0]}_{_run_sig1[-2]}_{_run_sig1[-1]}_{_sig_list[1]}_{_run_sig2[-2]}_{_run_sig1[-1]}'
            
        _sig_calc = _signal_JI if _signal_JI != 'CT2S' else 'T2S'
        _PSO_iter = pre_analysis.config.PSO_spec.PSO_math.cyc_pixel
        _n_comp   = pre_analysis.config.PSO_spec.PSO_math.n_comp
        _no_steps = pre_analysis.n_echoes[pre_analysis.decay_types[0]]
        _pix_list = pre_analysis.config.PSO_spec.comp_mode.PSO_on_pixel.pixel
        
        # random necessary for comparability of PSO runs with different parameters
        np.random.seed(0)
        max_workers = 56
        slice_size, slice_count = PSO_source.data.shape[:2], PSO_source.data.shape[2]

        # Log executor time
        start_time_cyc = time.time()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_base) as executor:
           
            results_dict = {f'{pixel[0]},{pixel[1]}': None for pixel in _pix_list}
            key          = str(_slice_calc[0]).zfill(2)
                
            for i, pix in enumerate(_pix_list):                
         
                start_time_pxl = time.time()
            
                # path specification
                save_path = os.path.normpath(os.path.join(PSO_source.file.prj_dir, 
                                                          'results', f'{_signal_JI}_JI', ID_string,
                                                          f'{str(PSO_spec.PSO_math.n_iter).zfill(3)}I'
                                                          f'{str(PSO_spec.PSO_math.n_part).zfill(3)}P'
                                                          f'{str(PSO_spec.PSO_math.cyc_pixel).zfill(2)}PSO',
                                                          f'{str(PSO_spec.PSO_math.n_comp)}_comp',
                                                          f'slice{key}', f'pixel_{pix[0]}_{pix[1]}'))

                os.makedirs(save_path, exist_ok=True)
                
                # definition of essential and execution parameters for each slice
                slice_param    = sys_param[key]
                np.random.seed(0)

                rand_vector    = list()
                rand_seeds     = np.random.randint(0, 100001, size=_PSO_iter)
                _, b1_idx_map  = _set_pix_param(_sig_list[:2], 0, slice_param, pre_analysis.data_type)
                
                # Memmaps für diesen Slice erstellen (in Slice-spez. Ordner)
                tmpdir = Path(os.path.join(save_path, "_mm")).resolve()
                tmpdir.mkdir(parents=True, exist_ok=True)

                # the first iteration pays the cost of making data visible; later iterations just reuse it
                # (writing mode "w+")
                sig1, sig2 = _sig_list[0], _sig_list[1]
                memdesc = {f'{sig1}_MATRIX':   _mk_memmap_from_array(slice_param[f'{sig1}_MATRIX'],  tmpdir, "f'{sig1}_MATRIX"),
                           f'{sig2}_MATRIX':   _mk_memmap_from_array(slice_param[f'{sig2}_MATRIX'],  tmpdir, "f'{sig2}_MATRIX"),
                           f'{sig1}_MASK':     _mk_memmap_from_array(slice_param[f'{sig1}_MASK'],    tmpdir, "f'{sig1}_MASK"),
                           f'{sig1}_B1_GRID':  _mk_memmap_from_array(slice_param[f'{sig1}_B1_GRID'], tmpdir, "f'{sig1}_B1_GRID"),
                           f'{sig2}_B1_GRID':  _mk_memmap_from_array(slice_param[f'{sig2}_B1_GRID'], tmpdir, "f'{sig2}_B1_GRID"),
                           f'{sig1}_MEAS_SIG': _mk_memmap_from_array(slice_param['MeasData'][sig1],  tmpdir, "f'{sig1}_MEAS_SIG"),
                           f'{sig2}_MEAS_SIG': _mk_memmap_from_array(slice_param['MeasData'][sig2],  tmpdir, "f'{sig2}_MEAS_SIG"),
                           'CT2S_TE':          _mk_memmap_from_array(slice_param['CT2S_TE'],    tmpdir, "CT2S_TE"),
                           'B1_DATA':          _mk_memmap_from_array(slice_param['B1_DATA'],    tmpdir, "B1_DATA"),}
                
                _broadcast_switch(executor    = executor,
                                  memdesc     = memdesc,
                                  b1_idx_map  = b1_idx_map,
                                  sig_list    = _sig_list,
                                  config      = config,
                                  constants   = pre_analysis.constants,
                                  max_workers = max_workers,)
            
                # time log for initialization of shared PSO resources for parallel workers
                if verbose and i == 0:
                    _PSO.log(start_time=start_time_pre, string='Execution time', dim='HMS')
                    print(f'\nRunning PSO joint inversion - pixel: {pix[0]},{pix[1]} | signal: {_signal_JI} | components: {_n_comp}')
                    
                # time log for preparation time of PSO algorithm
                if not verbose and i == 0:
                    _PSO.log(start_time=start_time_overall, string='Execution time', dim='HMS')
                    print(f'\nRunning PSO joint inversion - pixel: {pix[0]},{pix[1]} | signal: {_signal_JI} | components: {_n_comp}')
   
                # parallelization over a list of different randoms on same pixel
                # --> [x, y, rand_1], [x, y, rand_2], [x, y, rand_3]
                positions  = [[pix[0],pix[1],rand] for rand in rand_seeds]              
                results_sl = main_parallel_pixels(executor, positions, 1)
                results_dict[f'{pix[0]},{pix[1]}'] = results_sl                
                                
                # plot function pareto_pixel from PSOPlotter class
                pso_plotter.pareto_pixel_joint(inv_data=results_sl,pso_class=pre_analysis,num_val=10000,corridor=[True,3],
                                               save_path=f'{save_path}\\mwf_vs_misfit.png',save_format='PNG',save_dpi=300,save=True)
                
                # write and return results
                for kk, rand in enumerate(rand_seeds):
                    rand_vector.append((kk, rand))
                
                np.savetxt(f'{save_path}random_seeds.txt',
                           np.array(rand_vector, dtype=object), fmt="%s %s", delimiter=" ",
                           header="PSO_Iteration random_seed", comments="")
    
                _PSO.write_json(path=f'{save_path}')  
                  
                np.save(f'{save_path}\\results_asfromPSO.npy', results_sl, allow_pickle=True)
            
                _PSO.log(start_time=start_time_pxl, string='\nExecution time', dim='HMS')
                print(f'Saved results to {os.path.normpath(save_path)} ...')
     
            _PSO.log(start_time=start_time_cyc, string=f'\nExecution time {len(_pix_list)} Pixels:', dim='HMS')
            
            return results_dict, pre_analysis