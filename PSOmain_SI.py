# -*- coding: utf-8 -*-
"""
Main script for performing particle swarm optimization (PSO) on MRI invivo data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 02.2025; part of the JIMM/JIMM2 Project (DZNE Bonn & UFZ Leipzig)
"""

import os

# vor numpy/scipy imports
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

# # import of necessary tools: built-in, installed
import os, sys, time, concurrent.futures, glob
import numpy    as     np
import nibabel  as     nib
from   scipy.io import savemat

# # import of necessary tools: own scripts and classes
# # have to be stored in the same directory as the main script
import helpTools         as     hlp
from   PSOworkflow       import ParticleSwarmOptimizer as PSOclass
from   PSOpreparation_SI import PSOpreparation         as PSOprep
from   PSOplots          import PSOgrafics, PSOvideos

###############################################################################
# initializer function: runs one time for each worker
_G = {}

def _init_worker(sys_slice, config_static, constants, model_plot, sig_list, b1_index_map):
    _G['sys_param_slice'] = sys_slice
    _G['config']          = config_static
    _G['constants']       = constants
    _G['model_plot']      = model_plot
    _G['sig_list']        = sig_list
    _G['b1_idx_map']      = b1_index_map

from pathlib import Path

def _init_worker_base():
    _G.clear()
    _G['configured'] = False
    
def _mk_memmap_from_array(arr: np.ndarray, base_dir: str, name: str):
    """
    Schreibt arr C-kontigu nach base_dir/name.dat und liefert (path, shape, dtype, order).
    Wird nur im Hauptprozess benutzt.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = str(base_dir / f"{name}.dat")

    a = np.ascontiguousarray(arr)  # C-order sicherstellen
    mm = np.memmap(path, dtype=a.dtype, mode="w+", shape=a.shape)
    mm[:] = a[:]                   # write once
    del mm                         # flush & close
    return path, a.shape, a.dtype.str, "C"

def _open_memmap_ro(path: str, shape, dtype_str, order="C"):
    """
    Öffnet eine bestehende Memmap read-only im Worker.
    """
    return np.memmap(path, dtype=np.dtype(dtype_str), mode="r", shape=tuple(shape), order=order)

def _switch_slice(memdesc: dict, b1_index_map, sig_list, config_static, constants, model_plot):
    # wird im Worker ausgeführt
    _G['config']     = config_static
    _G['constants']  = constants
    _G['model_plot'] = model_plot
    _G['sig_list']   = sig_list
    _G['b1_idx_map'] = b1_index_map

    sig                 = sig_list[0]
    _G[f'{sig}_MATRIX'] = _open_memmap_ro(*memdesc[f'{sig}_MATRIX'])
    _G[f'{sig}_MASK']   = _open_memmap_ro(*memdesc[f'{sig}_MASK'])
    _G['CT2S_TE']       = _open_memmap_ro(*memdesc['CT2S_TE'])
    _G['B1_GRID']       = _open_memmap_ro(*memdesc['B1_GRID'])
    _G['B1_DATA']       = _open_memmap_ro(*memdesc['B1_DATA'])
    _G['MEAS_SIG']      = _open_memmap_ro(*memdesc['MEAS_SIG'])
    
    _G['configured']    = True

    import os
    return os.getpid()

def _broadcast_switch(executor, memdesc, b1_idx_map, sig_list, config, constants, model_plot, *, max_workers=None):
    # Anzahl Worker bestimmen (internes Attribut, aber praktisch)
    if max_workers is None:
        max_workers = getattr(executor, "_max_workers", None)
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    seen = set()
    # solange schicken, bis wirklich JEDER Worker einmal _switch_slice ausgeführt hat
    while len(seen) < max_workers:
        need = max_workers - len(seen)
        futs = [executor.submit(_switch_slice, memdesc, b1_idx_map, sig_list, config, constants, model_plot)
                for _ in range(need)]
        for f in concurrent.futures.as_completed(futs):
            seen.add(f.result())
            
###############################################################################
# parallelization over pixels within a slice ##################################
###############################################################################

def processSlice(position):

    if not _G.get('configured', False):
        raise RuntimeError("Worker nicht initialisiert. _broadcast_switch vor den Pixel-Tasks ausführen.")

    ##### getting variable input parameters from args
    yy, xx, rand = position

    ##### getting constant input parameters from initializer
    cfg_base   = _G['config']
    constants  = _G['constants']
    model_plot = _G['model_plot']
    sig_list   = _G['sig_list']
    b1_idx_map = _G['b1_idx_map']
    sig        = sig_list[0]
    
    sys_param_ = {
    f'{sig}_MATRIX': _G[f'{sig}_MATRIX'][:, :, b1_idx_map[yy, xx]],
    f'{sig}_MASK':   _G[f'{sig}_MASK'],
    'OBS_DATA':     {sig: _G['MEAS_SIG'][yy, xx, :]},
    'CT2S_TE':      _G['CT2S_TE'],
    }
    
    ##### adjusting dynamical parameters in config using flat copies
    config             = dict(cfg_base)
    pso_spec           = dict(config['PSO_spec'])
    dyn                = dict(pso_spec['dyn'])
    dyn['pixel']       = [yy, xx]
    dyn['rand']        = list(dyn['rand']); dyn['rand'][1] = rand
    pso_spec['dyn']    = dyn
    config['PSO_spec'] = pso_spec

    ##### PSO initalization
    PSO = PSOclass(config_data=config, sys_param=sys_param_, 
                   position=(yy,xx), init_matrix=True, 
                   constants=constants, model_plot=model_plot)

    try:
        ##### - termination condition for pixels, which turn out to show MWf = 0
        ##### - !!! memo: keep values from 0:19 and afterwards set everything to 0
        if sys_param_[f'{sig}_MASK'][position[0],position[1]] == 999:
            
            length   = PSO.noParam[sig]
            rel_time = getattr(PSO.Inversion, f"datSpace{sig}")

            return {'pix': [yy,xx], f'mod{sig}': np.full(length,np.nan), 
                    'fit': np.nan, 'synDat': np.full(rel_time,np.nan)}
    
        ##### execute PSO
        PSO.execPSO(obsData=sys_param_['OBS_DATA'], plotIterTest=False)    
        
        ##### return dictionary entry for each single pixel coordinate
        return {'pix': [yy,xx],  f'mod{sig}': PSO.globMod[sig],
                'fit': PSO.globFit[sig], 'synDat': PSO.globSynDat[sig]}
    
    finally:
        PSO._close()
        del PSO

###############################################################################
# parallelization over different randoms on a pixel ###########################
###############################################################################

def processPixel(position):

    if not _G.get('configured', False):
        raise RuntimeError("Worker nicht initialisiert. _broadcast_switch vor den Pixel-Tasks ausführen.")

    ##### getting variable input parameters from args
    yy, xx, rand = position

    ##### getting constant input parameters from initializer
    # sys_slice  = _G['sys_param_slice']
    cfg_base   = _G['config']
    constants  = _G['constants']
    model_plot = _G['model_plot']
    sig_list   = _G['sig_list']
    b1_idx_map = _G['b1_idx_map']
    sig        = sig_list[0]
    
    sys_param_ = {
    f'{sig}_MATRIX': _G[f'{sig}_MATRIX'][:, :, b1_idx_map[yy, xx]],
    f'{sig}_MASK':   _G[f'{sig}_MASK'],
    'OBS_DATA':     {sig: _G['MEAS_SIG'][yy, xx, :]},
    'CT2S_TE':      _G['CT2S_TE'],
    }
    
    ##### adjusting dynamical parameters in config using flat copies
    config             = dict(cfg_base)
    pso_spec           = dict(config['PSO_spec'])
    dyn                = dict(pso_spec['dyn'])
    dyn['pixel']       = [yy, xx]
    dyn['rand']        = list(dyn['rand']); dyn['rand'][1] = rand
    pso_spec['dyn']    = dyn
    config['PSO_spec'] = pso_spec

    ##### PSO initalization
    PSO = PSOclass(config_data=config, sys_param=sys_param_, 
                   position=(yy,xx), init_matrix=True, 
                   constants=constants, model_plot=model_plot)

    try:
        ##### return dictionary entry for each single pixel coordinate
        return {'pix': [yy,xx],  f'mod{sig}': PSO.globMod[sig],
                'fit': PSO.globFit[sig], 'synDat': PSO.globSynDat[sig]}
    
    finally:
        PSO._close()
        del PSO
        
###############################################################################
### creating parameters for feeding the parallelized algorithm with ###########
###############################################################################

def set_pix_param(sig: str, rand: int, slice_key: int, sys_param: dict, data_type='invivo'):
    
    key         = str(slice_key).zfill(2)
    slice_param = sys_param[key]
    
    B1_GRID     = slice_param['B1_GRID']
    B1_DATA     = slice_param['B1_DATA']
    
    mask        = slice_param[f'{sig}_MASK']  
    coords      = np.argwhere(mask == 1)
    
    positions   = [(int(y), int(x), int(rand)) for y, x in coords]
    
    if data_type == 'atlas':
        return positions, np.zeros_like(B1_DATA, dtype=np.int16)

    if data_type == 'invivo':
        b1_idx_map  = np.empty_like(B1_DATA, dtype=np.int16)
        
        for (yy, xx), _ in np.ndenumerate(B1_DATA):
            b1_idx_map[yy, xx] = np.argmin(np.abs(B1_GRID - B1_DATA[yy, xx]))
    
        return positions, b1_idx_map

###############################################################################
### main function for parallelization, which calls process subfunctionality ###
###############################################################################

def main_parallel_slice_static(executor: object, pix_param: list, chunksize=1):
    return list(executor.map(processSlice, pix_param, chunksize=chunksize))

def main_parallel_pixel_static(executor: object, pix_param: list, chunksize=1):    
    return list(executor.map(processPixel, pix_param, chunksize=chunksize))

def dummy(x):
    return x

###############################################################################
### break up criteria for certain pixels ###
###############################################################################

def enhance_mask(PSO_result: np.array, mask: np.array, signal: str):
    for item in PSO_result:
        y, x = item['pix']
        mask[y, x] += item[f'mod{signal}'][-1]
    return mask

###############################################################################
### PSO execution area: different modes possible, depending on Users input ####
###############################################################################

def run_PSO_on_config(config_input: str, config_type: str):
    
    # user section:
    if config_type == 'file':
        config_path = config_input
        config      = hlp.open_parameters(path=config_path)
    if config_type == 'dict':
        config      = config_input

    # troubleshooting - controll if main parameters are set right
    config      = hlp.troubleshooting(config=config)

    # global parameter calculation application 
    # PSO class initialization to gain access to maiden methods/objects
    _PSO        = PSOclass(config_data=config)
    _PSOgrafics = PSOgrafics()

    # preparing PSO analysis
    for o in range(1):
        if config['PSO_spec']['calc']['iter_test'] == True:
            _slice_calc  = list([config['PSO_spec']['calc']['iter_test']['slice']])
            _slice_calc.append(_slice_calc[0]+1); continue
        if config['PSO_spec']['calc']['performance_test']['use'] == True:
            _slice_calc  = [7,8]; continue
        if config['PSO_spec']['calc']['PSO_on_pixel']["use"] == True:
            _slice_calc  = list([config['PSO_spec']['calc']['PSO_on_pixel']['slice']])
            _slice_calc.append(_slice_calc[0]+1); continue
        if config['PSO_spec']['calc']['PSO_on_slice']["use"] == True:
            _slice_start = config['PSO_spec']['calc']['PSO_on_slice']['start']
            _slice_end   = config['PSO_spec']['calc']['PSO_on_slice']['end']
            _slice_calc  = list(range(_slice_start, _slice_end+1))
    
    pre_analysis = PSOprep(config_input = config_input,
                           config_type  = config_type,
                           mask_use     = config['source']['mask']['use'],
                           mask_path    = config['source']['mask']['path'],
                           data_type    = config['source']['data']['obs_data'])
    
    pre_analysis.calculate_sys_param(slice_calc=_slice_calc)  
    sys_param    = pre_analysis.sys_param

    # defining variables
    _sig_list   = [k for k in list(config['source']['signal'].keys())[:4] if config['source']['signal'][k]]
    _signal_SI  = _sig_list[0] if _PSO.invCT2S == False else 'CT2S'
        
    # time logging    
    gmtTime          = time.strftime('%H:%M:%S', time.gmtime())
    print(f'\nStarttime for PSO routine (GMT): {gmtTime}')

###############################################################################       
### (a) Performance test: #####################################################
###     --> how many iterations until diff: fit(n)-fit(n-1) < thresh ##########
    
    if pre_analysis.config.PSO_spec.calc.iter_test.use == True:

        _save_path_ID = 'results/performance_test/iteration_test'
        _norm_max     = pre_analysis.config.source.data.norm_max
        _filt_gauss   = pre_analysis.config.source.data.gauss_filt[1]
        _pixel_list   = pre_analysis.config.PSO_spec.calc.iter_test.pixli
        _create_gif   = pre_analysis.config.PSO_spec.calc.iter_test.gif
        _slice        = str(pre_analysis.config.PSO_spec.calc.iter_test.slice).zfill(2)
        _mask_list    = list()
        
        project_dir = pre_analysis.config.source.file.prj_dir
        savepathPSO = os.path.join(project_dir, _save_path_ID)
        comp_slice  = os.path.join(project_dir, f'results/raw_data/run1/100I400P05PSO/{_sig_list[0]}/GAUSS/Slice{_slice}/')

        for ii, item in enumerate(_pixel_list[:1]):

            yy, xx = item[0], item[1]
            
            print(f'... running PSO for pixel [{yy},{xx}]')
            config['PSO_spec']['dyn']['pixel']   = [yy, xx]
            config['PSO_spec']['dyn']['rand'][1] = 0
            config['PSO_spec']['dyn']['slice']   = int(_slice)
            
            n_fs       = np.argmin(np.abs(sys_param[_slice]['B1_GRID']-sys_param[_slice]['B1_DATA'][yy,xx]))
            sys_param_ = {'T2_MATRIX':  sys_param[_slice]['T2_MATRIX'][:,:,n_fs], 
                          'T2S_MATRIX': sys_param[_slice]['T2S_MATRIX'][:,:,0],
                          'MASK':       sys_param[_slice]['MASK'],
                          'CT2S_TE':    sys_param[_slice]['CT2S_TE']}

            # PSO class instance
            PSO = PSOclass(config_data=config, sys_param=sys_param_, 
                           position=(yy,xx), init_matrix=True, model_plot=(True, savepathPSO),
                           constants=pre_analysis.constants)    
            
            # getting the observed data per pixel of interest
            obsData  = PSO.get_data_per_pixel(position   = (yy,xx,int(_slice)),
                                              data_dir   = os.path.normpath(f'{config["source"]["file"]["prj_dir"]}/nifti/'),
                                              norm_max   = _norm_max,
                                              filt_gauss = _filt_gauss,
                                              pathT1     = config["source"]["file"]['T1'],
                                              pathT2     = config["source"]["file"]['T2'],
                                              pathT2S    = config["source"]["file"]['T2S'],
                                              pathT2SP   = config["source"]["file"]['CT2S'],
                                              pathTE     = config["source"]["file"]['TE'])

            # fully calculated brain for comparison
            directory = f'F:/JIMM2/MWF_invivo/DZNE_Data/24_10_10_c/results/raw_data/run1/100I400P05PSO/{_sig_list[0]}/GAUSS/Slice{_slice}/'
           
            MWFdataRaw = nib.load(os.path.normpath(f'{config["source"]["file"]["prj_dir"]}/nifti/{config["source"]["file"]["T2S"]}'))
            MWFdataRaw = MWFdataRaw.get_fdata()
            MWFdataRaw = np.sum(MWFdataRaw, axis=3)[:,:,int(_slice)]
                        
            # execute PSO for one pixel of interest
            PSO.execPSO(obsData=obsData,plotIterTest=True,
                        callback_plot=lambda **kwargs: _PSOgrafics.plotIterTest(
                        MWFdata=MWFdataRaw, PSOclass=kwargs["PSO"], position=(yy,xx,int(_slice)),
                        savepath=savepathPSO, string=kwargs["string"]))
            
            # filling the mask and calculating when 
            percentThresh = 0.1
            aa   = np.abs(np.diff(PSO.globFit_list[_signal_SI]) / PSO.globFit_list[_signal_SI][:-1] * 100)
            mask = np.abs(aa) >= percentThresh
            _mask_list.append(np.where(mask)[0][-1])
            
            def create_gif(pixel: list, directory='', fps=1, loops=0):
                
                ''' Creates a gif from files in a given filelist. '''
                
                for pixel in pixel:
 
                    yy, xx    = pixel[0], pixel[1]
                    directory = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\results\performance_test\iteration_test\T2'
                    savepath  = os.path.join(directory, f'{yy}_{xx}.gif')
                    filelist  = glob.glob(os.path.join(directory, '*iter*.jpg'))
                    filelist  = [f for f in filelist if f'y-{yy}_x-{xx}' in f]
                    
                    PSOmovies = PSOvideos()
                    
                    PSOmovies.build_gif(filelist=filelist, savepath=savepath, fps=fps, loops=loops)
            
            if _create_gif == True:
                create_gif(_pixel_list, fps=2, loops=5)
            
            _PSOgrafics.plotIterTest(MWFdata=MWFdataRaw, PSOclass=PSO, position=(yy,xx,int(_slice)),
                                     savepath=savepathPSO, string=f'{_signal_SI}')

            print(f'Number of iterations before threshold is reached: {np.where(mask)[0][-1]}\n')
            
        sys.exit(f'Iteration test finished. Worst mask ID {np.max(_mask_list)}')

###############################################################################    
### (b.1) Parallelized calculation of MWF for one slice and with n PSO cycles
###       invOpt V0 --> Each signal has a respective model vector. 
###                     MWF is present in the objective function.
###      
###       invOpt V1 --> One model vector for both signals combined (same MWF).
###                     MWF is not present in the objective function.

    if pre_analysis.config.PSO_spec.calc.PSO_on_slice.use==True:

        brain_anatomy = ['CSF', 'GM', 'WM', 'dGM', 'BS', 'CB']
        
        dat = 'raw_data' if pre_analysis.config.source.data.gauss_filt[0] == False else 'filt_data'
        ana = "_".join([brain_anatomy[i] for i in [x - 1 for x in pre_analysis.config.source.mask.seg]])
        run = os.path.basename(pre_analysis.config.source.file.T2).removeprefix('mese_').removesuffix('.gz').removesuffix('.nii')

        print(run)
        
        # run = os.path.basename(pre_analysis.config.source.file.T2).split('.')[0].split('_')[-1]
        
        ID_string   = os.path.join(dat, run, ana)
        
        _save_dir = os.path.normpath(os.path.join(f'{config["source"]["file"]["prj_dir"]}', 'results', f'{ID_string}',
                                                  f'{str(config["PSO_spec"]["iter"]).zfill(3)}I'
                                                  f'{str(config["PSO_spec"]["part"]).zfill(3)}P'
                                                  f'{str(config["PSO_spec"]["PSO_iter"]["slice"]).zfill(2)}PSO',
                                                  f'{_signal_SI}', f'{config["PSO_spec"]["peaks"]}'))
        
        _sig_calc = _signal_SI if _signal_SI != 'CT2S' else 'T2S'
        
        _PSO_iter = pre_analysis.config.PSO_spec.PSO_iter.slice
        _no_peaks = pre_analysis.config.PSO_spec.peaks
        _no_param = pre_analysis.noParam[_sig_calc] + 1
        _no_steps = pre_analysis.noSteps[pre_analysis.signType[0]]
        os.makedirs(_save_dir, exist_ok=True)
        
        # random necessary for comparability of PSO runs with different parameters
        np.random.seed(0)
        startTime   = time.time()
        max_workers = 56
        slice_count = 24 if pre_analysis.data_type == 'invivo' else 182
        slice_size  = [110,110] if pre_analysis.data_type == 'invivo' else [182,218]

        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_base) as executor:
            
            for i, item in enumerate(range(slice_count)[_slice_start:_slice_end]):
    
                if pre_analysis.config.PSO_spec.calc.performance_test.use == True: continue

                # definition of essential and execution parameters for each slice
                key             = str(item).zfill(2)
                slice_param     = sys_param[key]
                
                startTime_slice = time.time()
                rand_vector     = list()
                savepathPSO     = f'{_save_dir}\\Slice{str(item).zfill(2)}\\'               
                
                result_map_sl   = {sig: np.zeros([_no_param, slice_size[0], slice_size[1], _PSO_iter+1]) for sig in _sig_list[:1]}
                syn_dat_map     = {_sig_calc: np.empty((slice_size[0], slice_size[1], _no_steps, _PSO_iter))}
                res_array_dic   = np.empty((slice_size[0], slice_size[1], _PSO_iter), dtype=object)
                
                mask0                                 = slice_param[f'{_sig_calc}_MASK'] == 0
                result_map_sl[_sig_calc][:, mask0, :] = np.nan
                syn_dat_map[_sig_calc][mask0, :, :]   = np.nan
                rand_seeds                            = np.random.randint(0, 100001, size=_PSO_iter)
                _, b1_idx_map                         = set_pix_param(_sig_calc, 0, item, sys_param, pre_analysis.data_type)
                
                # Memmaps für diesen Slice erstellen (in Slice-spez. Ordner)
                tmpdir = Path(os.path.join(_save_dir, f"Slice{key}", "_mm")).resolve()
                tmpdir.mkdir(parents=True, exist_ok=True)
   
                memdesc = {f'{_sig_calc}_MATRIX':  _mk_memmap_from_array(slice_param[f'{_sig_calc}_MATRIX'],  tmpdir, "f'{_sig_calc}_MATRIX"),
                           f'{_sig_calc}_MASK':    _mk_memmap_from_array(slice_param[f'{_sig_calc}_MASK'],    tmpdir, "f'{_sig_calc}_MASK"),
                           'CT2S_TE':              _mk_memmap_from_array(slice_param['CT2S_TE'],    tmpdir, "CT2S_TE"),
                           'B1_GRID':              _mk_memmap_from_array(slice_param['B1_GRID'],    tmpdir, "B1_GRID"),
                           'B1_DATA':              _mk_memmap_from_array(slice_param['B1_DATA'],    tmpdir, "B1_DATA"),
                           'MEAS_SIG':             _mk_memmap_from_array(slice_param['MeasData'][_sig_calc], tmpdir, "MEAS_SIG"),}
                
                _broadcast_switch(executor    = executor,
                                  memdesc     = memdesc,
                                  b1_idx_map  = b1_idx_map,
                                  sig_list    = _sig_list[:1],
                                  config      = config,
                                  constants   = pre_analysis.constants,
                                  model_plot  = (False, ""),
                                  max_workers = max_workers,)
                
                _PSO.log(startTime=startTime_slice, string='\nExecution time preparation of sys param: ', dim='HMS')

                print(f'\nExecute PSO inversion - signal: {_signal_SI} | slice: {key} | noPeaks: {_no_peaks}')
                
                for kk, rand in enumerate(rand_seeds):
                    
                    #startTime_rand = time.time()
                    
                    #if kk in [1,2,99]:
                        #print(f'\nExecute PSO {str(kk).zfill(2)} - signal: {_signal_SI} | slice: {key} | noPeaks: {_no_peaks}')
                    rand_vector.append((f'{str(kk).zfill(2)}', rand))
                    
                    positions, _ = set_pix_param(_sig_calc, rand, item, sys_param)
                    results_sl   = main_parallel_slice_static(executor, positions, 1)
                    
                    result_map_sl, syn_dat_map, res_array_dic = _PSO.result2array(
                                                                     results_sl, result_map_sl, syn_dat_map, res_array_dic, kk,
                                                                     cutThresh      = (None,None,False), 
                                                                     cutMask        = (sys_param[key][f'{_sig_calc}_MASK'],False),
                                                                     calcBestResult = True,
                                                                     arrayType      = 'Slice')
            
                    #if kk in [0,1,2,99]:
                    #    _PSO.log(startTime=startTime_rand, string=f'\nExecution time rand {str(kk).zfill(2)}: ', dim='ms')
                
                    #_PSOgrafics.plotSlice(PSOclass=_PSO, PSOresult=result_map_sl, index=kk,
                    #                      saveFig=(savepathPSO,True), string='')

                    #_PSO.log(startTime=startTime_rand, string=f'\nExecution time slice {str(item).zfill(2)}: ', dim='HMS')
    
                    if kk==_PSO_iter-1:
                        _PSOgrafics.plotSlice(PSOclass=_PSO, PSOresult=result_map_sl, index=-1,
                                              saveFig=(savepathPSO,True), string='')

                np.savetxt(f'{savepathPSO}random_seeds.txt',
                           np.array(rand_vector, dtype=object), fmt="%s %s", delimiter=" ",
                           header="PSO_Iteration random_seed", comments="")
    
                np.save(f'{savepathPSO}{_sig_calc}_results_asfromPSO.npy',   res_array_dic, allow_pickle=True)
                
                nib.save(nib.Nifti1Image(result_map_sl[_sig_calc], affine=np.eye(4)), 
                         f'{savepathPSO}{_sig_calc}_results_modelvector.nii.gz')                
                np.save(f'{savepathPSO}{_sig_calc}_results_modelvector.npy', result_map_sl, allow_pickle=True)
                
                nib.save(nib.Nifti1Image(syn_dat_map[_sig_calc], affine=np.eye(4)),
                         f'{savepathPSO}{_sig_calc}_results_syndata.nii.gz')
                np.save(f'{savepathPSO}{_sig_calc}_results_syndata.npy',     syn_dat_map,   allow_pickle=True)
                
                
                savemat(f'{savepathPSO}{_sig_calc}_results_modelvector.mat', result_map_sl, do_compression=True)
                savemat(f'{savepathPSO}{_sig_calc}_results_syndata.mat',     syn_dat_map, do_compression=True)
                
                _PSO.write_JSON(savepath=f'{savepathPSO}')

                _PSO.log(startTime=startTime_slice, string=f'\nExecution time slice {str(item).zfill(2)}: ', dim='HMS')
                
                # del result_map_sl, results_sl, positions, help_mask, syn_dat_map, res_array_dic


            for i, item in enumerate(range(24)[7:8]):

                if pre_analysis.config.PSO_spec.calc.performance_test.use == False: continue
        
                key         = str(item).zfill(2)
                slice_param = sys_param[key]
                project_dir = config["source"]["file"]["prj_dir"]
                savepathID  = 'MSG\\v_200_4'                
                savepathPSO = f'{project_dir}\\results\\performance_test\\{savepathID}\\'                                       
                os.makedirs(savepathPSO, exist_ok=True)
            
                _, b1_idx_map   = set_pix_param(0, item, sys_param)
                
                tmpdir = Path(os.path.join(savepathPSO, "_mm")).resolve()
                tmpdir.mkdir(parents=True, exist_ok=True)
    
                memdesc = {'T2_MATRIX':  _mk_memmap_from_array(slice_param['T2_MATRIX'],  tmpdir, "T2_MATRIX"),
                           'T2S_MATRIX': _mk_memmap_from_array(slice_param['T2S_MATRIX'], tmpdir, "T2S_MATRIX"),
                           'MASK':       _mk_memmap_from_array(slice_param['MASK'],       tmpdir, "MASK"),
                           'CT2S_TE':    _mk_memmap_from_array(slice_param['CT2S_TE'],    tmpdir, "CT2S_TE"),
                           'B1_GRID':    _mk_memmap_from_array(slice_param['B1_GRID'],    tmpdir, "B1_GRID"),
                           'B1_DATA':    _mk_memmap_from_array(slice_param['B1_DATA'],    tmpdir, "B1_DATA"),}
                
                for kk, rand in enumerate(np.random.randint(0, 100001, size=1)):
                    
                    for _no_peaks in ['GAUSS', 'DIRAC'][:1]:
                        
                        config["PSO_spec"]["peaks"] = _no_peaks
                        config["PSO_spec"]["PSO_iter"]["slice"] = 1
                        
                        for i, _sig_calc in enumerate(['T2', 'T2S', 'CT2S'][2:]):
    
                            _sig_sig = _sig_calc if _sig_calc != 'CT2S' else 'T2S'
                            # sigma        = getattr(getattr(pre_analysis, _sig_calc), _no_peaks).m1_sig[-1]
                            sigma        = getattr(getattr(pre_analysis, _sig_sig), _no_peaks).m1_sig[0]
                            string       = f'{_no_peaks}_{_sig_calc}_{sigma}'                        
                            _savepathPSO = f'{savepathPSO}{string}'
                            
                            # signalflags
                            config['source']['signal']['T2']   = (_sig_calc == 'T2')
                            config['source']['signal']['T2S']  = (_sig_calc in ('T2S','CT2S'))
                            config['source']['signal']['CT2S'] = (_sig_calc == 'CT2S')
                            
                            # run PSO again for metadata
                            _PSO      = PSOclass(config_data=config)
                            _no_steps = _PSO.noSteps[_PSO.signType[0]]
                            _PSO_iter = 1
                            
                            # set measured signal for specific data signal
                            _filter = pre_analysis.config.source.data.gauss_filt[0]
                            
                            pre_analysis.get_obs_data(_filter_bool=_filter, _filter_vals=(1.0,1.0,0), _signal=_sig_calc)
                            _obs_data = pre_analysis.obs_data[_sig_calc][:, :, item, :]
                            memdesc['MEAS_SIG'] = _mk_memmap_from_array(_obs_data, tmpdir, f"MEAS_{_sig_calc}")
                            
                            # sys_param[key]['MeasData'][_sig_calc]=pre_analysis.obs_data[_sig_calc][:,:,item,:]
                            
                            _broadcast_switch(executor    = executor,
                                              memdesc     = memdesc,
                                              b1_idx_map  = b1_idx_map,
                                              sig_list    = [_sig_calc],
                                              config      = config,
                                              constants   = pre_analysis.constants,
                                              model_plot  = (False, ""),
                                              max_workers = max_workers,)
                            
                            if _sig_calc == 'T2':
                                result_map_sl = {'T2': np.zeros([_PSO.noParam[_sig_calc]+1, 110, 110, _PSO_iter+1])}
                                syn_dat_map   = {'T2': np.empty((110, 110, _no_steps, _PSO_iter))}
                                res_array_dic = np.empty((110, 110, _PSO_iter), dtype=object)
                            else:
                                result_map_sl = {'T2S': np.zeros([_PSO.noParam[_sig_calc]+1, 110, 110, _PSO_iter+1])}
                                syn_dat_map   = {'T2S': np.empty((110, 110, _no_steps, _PSO_iter))}
                                res_array_dic = np.empty((110, 110, _PSO_iter), dtype=object)
    
                            mask0 = slice_param['MASK'] == 0
                            result_map_sl[_sig_sig][:, mask0, :] = np.nan
                            syn_dat_map[_sig_sig][mask0, :, :]   = np.nan

                            startTime_slice = time.time()
                            
                            print(f'\nExecute PSO {str(kk).zfill(2)} - signal: {_sig_calc} | slice: {str(item).zfill(2)} | noPeaks: {_no_peaks}')  
                            
                            positions, _ = set_pix_param(rand, item, sys_param)                  
                            results_sl   = main_parallel_slice_static(executor, positions, 1)
        
                            result_map_sl, syn_dat_map, res_array_dic = _PSO.result2array(
                                                                        results_sl, result_map_sl, syn_dat_map, res_array_dic, kk,
                                                                        cutThresh      = (None,None,False), 
                                                                        cutMask        = (sys_param[key]['MASK'],False),
                                                                        calcBestResult = True,
                                                                        arrayType      = 'Slice')
                    
                            _PSO.log(startTime=startTime_slice, string=f'\nExecution time slice {str(item).zfill(2)} ({_sig_calc}): ', dim='HMS')
    
                            _PSOgrafics.plotSlice(PSOclass=_PSO, PSOresult=result_map_sl, index=kk, 
                                                  saveFig=(_savepathPSO,True), string='', performance_test=True)
    
                            np.save(f'{_savepathPSO}_results.npy', result_map_sl)
                            savemat(f'{_savepathPSO}_results.mat', result_map_sl)

        _PSO.log(startTime=startTime, string='\nExecution time all together: ', dim='HMS')
        
        return syn_dat_map, result_map_sl, pre_analysis
    
#########################################################################################################

### (b.2) Parallelized calculation of MWF for one pixel with different randoms

    if pre_analysis.config.PSO_spec.calc.PSO_on_pixel.use==True:
        
        pix  = pre_analysis.config.PSO_spec.calc.PSO_on_pixel.pixel
        dat  = 'raw_data' if pre_analysis.config.source.data.gauss_filt[0] == False else 'filt_data'
        run  = os.path.basename(pre_analysis.config.source.file.T2).split('.')[0].split('_')[-1]
        SNR  = pre_analysis.config.source.data.add_noise[1] if pre_analysis.config.source.data.add_noise[0] == True else 0
        norm = '_normed' if pre_analysis.config.PSO_spec.math.norm == True else ''
        
        ID_string  = os.path.join(dat, run, f'pixel_{pix[0]}_{pix[1]}_SNR{SNR}{norm}')
        
        _save_dir = os.path.normpath(os.path.join(f'{config["source"]["file"]["prj_dir"]}', 'results', f'{ID_string}',
                                                  f'{str(config["PSO_spec"]["iter"]).zfill(3)}I'
                                                  f'{str(config["PSO_spec"]["part"]).zfill(3)}P'
                                                  f'{str(config["PSO_spec"]["PSO_iter"]["pixel"]).zfill(2)}PSO',
                                                  f'{_signal_SI}', f'{config["PSO_spec"]["peaks"]}'))
        
        _sig_calc = _signal_SI if _signal_SI != 'CT2S' else 'T2S'
        _PSO_iter = pre_analysis.config.PSO_spec.PSO_iter.pixel
        _no_peaks = pre_analysis.config.PSO_spec.peaks
        _no_param = pre_analysis.noParam[_sig_calc] + 1
        _no_steps = pre_analysis.noSteps[pre_analysis.signType[0]]
        _slice    = pre_analysis.config.PSO_spec.calc.PSO_on_pixel.slice
        os.makedirs(_save_dir, exist_ok=True)
        
        # random necessary for comparability of PSO runs with different parameters
        np.random.seed(0)
        startTime = time.time()
        max_workers = 56

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_base) as executor:
            
            # definition of essential and execution parameters for each slice
            key             = str(_slice).zfill(2)
            slice_param     = sys_param[key]
            
            startTime_slice = time.time()
            savepathPSO     = f'{_save_dir}\\Slice{key}\\'  
            
            result_map_sl   = {sig: np.empty([_no_param, _PSO_iter]) for sig in _sig_list[:1]}
            syn_dat_map     = {_sig_calc: np.empty((_no_steps, _PSO_iter))}
            res_array_dic   = np.empty(_PSO_iter, dtype=object)
            
            rand_seeds      = np.random.randint(0, 100001, size=_PSO_iter)
            _, b1_idx_map   = set_pix_param(_sig_calc, 0, _slice, sys_param, pre_analysis.data_type)
            
            # Memmaps für diesen Slice erstellen (in Slice-spez. Ordner)
            tmpdir = Path(os.path.join(_save_dir, f"Slice{key}", "_mm")).resolve()
            tmpdir.mkdir(parents=True, exist_ok=True)

            memdesc = {f'{_sig_calc}_MATRIX':  _mk_memmap_from_array(slice_param[f'{_sig_calc}_MATRIX'],  tmpdir, "f'{_sig_calc}_MATRIX"),
                       f'{_sig_calc}_MASK':    _mk_memmap_from_array(slice_param[f'{_sig_calc}_MASK'],    tmpdir, "f'{_sig_calc}_MASK"),
                       'CT2S_TE':              _mk_memmap_from_array(slice_param['CT2S_TE'],    tmpdir, "CT2S_TE"),
                       'B1_GRID':              _mk_memmap_from_array(slice_param['B1_GRID'],    tmpdir, "B1_GRID"),
                       'B1_DATA':              _mk_memmap_from_array(slice_param['B1_DATA'],    tmpdir, "B1_DATA"),
                       'MEAS_SIG':             _mk_memmap_from_array(slice_param['MeasData'][_sig_calc], tmpdir, "MEAS_SIG"),}
            
            _broadcast_switch(executor    = executor,
                              memdesc     = memdesc,
                              b1_idx_map  = b1_idx_map,
                              sig_list    = [_sig_calc],
                              config      = config,
                              constants   = pre_analysis.constants,
                              model_plot  = (False, ""),
                              max_workers = max_workers,)
            
            _PSO.log(startTime=startTime_slice, string='\nExecution time preparation of sys param: ', dim='HMS')
                
            startTime_pixel = time.time()
            
            print(f'\nExecute PSO on pixel [{pix[0]},{pix[1]}]: signal - {_sig_calc} | slice: {key} | noPeaks: {_no_peaks}')
            
            positions  = [[pix[0],pix[1],rand] for rand in rand_seeds]                
            results_sl = main_parallel_slice_static(executor, positions, 1)
            
            # 0: m1 | 1: m1sig | 2: m2 | 3: m2sig | 4: intFW | 5: MWF | 6,7,8: MW_f, FW_f, phi
            MWF        = np.array([r[f"mod{_sig_calc}"][5] for r in results_sl])
            fits       = np.array([r["fit"]                for r in results_sl])

            idx_sorted = np.argsort(fits)
            idx_best   = idx_sorted[:10000]
            idx_worst  = idx_sorted[10000:]
            
            import matplotlib.pyplot as plt
            
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
            
            # --- Subplot a1 ---
            a1.plot(fits[idx_best]*1e3, MWF[idx_best], markersize=2, linestyle='none', marker='o',
                    markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                    label='10e3 best MWF values')
            
            a1.plot(fits[idx_worst]*1e3, MWF[idx_worst], markersize=2, linestyle='none', marker='o',
                    markeredgewidth=0.5, color='gray', markerfacecolor='lightgray',
                    label='')            
            
            a1.plot(np.min(fits)*1e3, MWF[fits==np.min(fits)][0],
                    markersize=3, linestyle='none', marker='o', markeredgewidth=0.5, 
                    color='b', label=f'bestMWF: {np.round(MWF[fits==np.min(fits)][0],3)}')

            if pre_analysis.data_type == 'atlas':
                x, y           = pre_analysis.config.PSO_spec.calc.PSO_on_pixel.pixel
                MWF_from_atlas = pre_analysis.obs_data['MWF_ATLAS'][x,y,_slice]
               
                a1.plot(np.min(fits[idx_best])*1e3, MWF_from_atlas,
                        markersize=0, linestyle='none', marker='o', markeredgewidth=0.5,
                        color='b', label=f'atlas MWF: {np.round(MWF_from_atlas,3)}')
            
            low, high = np.min(fits)*1e3, np.min(fits)*1e3*1.03
            a1.axvspan(low, high, color="lightblue", alpha=0.5)
            
            # Dummy-Patch für Legende
            a1.plot([], [], color="lightblue", linewidth=8, alpha=0.5,
                    label='corridor: bestFit +3% * BF')
            
            a1.legend(loc='upper right')
            a1.set_xlabel(r'global best Fit [$\times 10^{-3}$]')
            a1.set_ylabel('global best MWF []')
            
            a2.plot(fits[idx_best]*1e3, MWF[idx_best], markersize=2, linestyle='none', marker='o',
                    markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
                    label='10e3 best MWF values')
            
            MWF_of_best_fit = MWF[idx_sorted[np.argmin(fits[idx_sorted])]]
            
            a2.plot(np.min(fits[idx_best])*1e3, MWF_of_best_fit,
                    markersize=3, linestyle='none', marker='o', markeredgewidth=0.5,
                    color='b', label=f'best MWF: {np.round(MWF_of_best_fit,3)}')
            
            # a2.axvspan(low, high, color="lightblue", alpha=0.5)
            
            # ymin, ymax = a1.get_ylim()
            # a2.set_ylim(ymin, ymax)
            
            a2.legend(loc='upper right')
            a2.set_xlabel('global best Fit []')
            a2.set_ylabel('global best MWF []')

            a2.set_xlabel(r'global best Fit [$\times 10^{-3}$]')
            
            plt.tight_layout()
            plt.show()
            
            _PSO.log(startTime=startTime_pixel, string='\nExecution time', dim='HMS')

            fig.savefig(f'{_save_dir}/MWF_vs_FIT.png', dpi=300, format='png', bbox_inches='tight')
            
            for j, line in enumerate(results_sl):
                res_array_dic[j] = line
            
            np.save(f'{_save_dir}/results_asfromPSO.npy', res_array_dic, allow_pickle=True)
            
            return results_sl, pre_analysis
        
### Return result arrays for further use in e.g. jupyter notebook
    #return syn_dat_map, result_map_sl, pre_analysis
    
#         result_map_px       = {sig: np.zeros([7,  noPSOIterPix]) if noPeaks == 'GAUSS' 
#                                else np.zeros([10, noPSOIterPix]) for sig in typeSig}

#         signal              = ('').join([sig for sig in typeSig])        
        
#         for kk, (yy,xx) in enumerate(pixelList):

#             print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {noPeaks}')      

#             results       = main_parallel_pixel(yy,xx,modParam=('class', noPeaks, None))
            
                # result_map_sl, syn_dat_map, res_array_dic = _PSO.result2array(
                #                                             results_sl, result_map_sl, syn_dat_map, res_array_dic, 0,
                #                                             cutThresh      = (None,None,False), 
                #                                             cutMask        = (sys_param[key]['MASK'],False),
                #                                             calcBestResult = True,
                #                                             arrayType      = 'Slice')
            
#             _PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map_px,
#                                      PSOresultSLI=result_map_sl,position=(yy,xx,noSlice), 
#                                      saveFig=(f'{savepathPSO}{noPeaks}/',True),
#                                      string=f'MWFvsFit_{noPSOIterPix}Iter',
#                                      valOutliers=(5,0.1),cutOutliers=True,  
#                                      valPercentile=(0,50),cutPercentile=True)

#             yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
#             savepath = f'{savepathPSO}{noPeaks}/MWFvsFit_{noPSOIterPix}Iter/'            
#             np.save(f'{savepath}y{yy}x{xx}_pixelresult.npy', result_map_px)
#             PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

#########################################################################################################

# ### (b.3) Parallelized calculation of MWF for one pixel with different randoms

#     if calcPSO_pixel==True and calcPSO_slice==False:
        
#         for file in resultsList:
            
#             print(f'\nCurrent file (PSO CALC PIXEL):\n{file}')
            
#             # get parameters from npy and json files
#             PSO.signType = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
#             PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
#             signal       = ('').join([sig for sig in PSO.signType])
            
#             result_map_sl = np.load(file, allow_pickle=True).item()
#             result_map_px = {sig: np.zeros([7,  noPSOIterPix]) if PSO.noPeaks == 'GAUSS' 
#                              else np.zeros([10, noPSOIterPix]) for sig in PSO.signType}
    
#             dirPA  = os.path.dirname(file)
#             pathPA = os.path.join(dirPA, f'{signal}_param.json')
                                  
#             with open(pathPA, 'r') as json_file: 
#                 jsonParam     = json.load(json_file)
#                 modParam      = ('json', PSO.noPeaks, jsonParam)
#                 PSO.noIter    = modParam[-1]['PSO specifications']['Iterations']
#                 PSO.noPart    = modParam[-1]['PSO specifications']['Particles']
#                 PSO.noPSOIter = noPSOIterPix
            
#             np.random.seed(0)
            
#             for kk, (yy,xx) in enumerate(pixelList):

#                 pixelpath = os.path.join(dirPA, f'MWFvsFIT_{noPSOIterPix}Iter', f'y{yy}x{xx}_pixelresult.npy')

#                 if os.path.exists(pixelpath):
#                     print(f'Already existing: pixel result for y{yy}x{xx}. Continue...')
#                     continue
                
#                 print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {PSO.noPeaks}')
                
#                 results       = main_parallel_pixel(yy,xx,modParam=modParam)
                
#                 result_map_px = PSO.result2array(results, result_map_px, kk,
#                                                  cutThresh      =(None,None,False), 
#                                                  cutMask        =(resultsMask,False),
#                                                  calcBestResult =False,
#                                                  arrayType      ='Pixel')        
        
#                 _PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map_px,
#                                          PSOresultSLI=result_map_sl,position=(yy,xx,noSlice), 
#                                          saveFig=(f'{os.path.dirname(file)}/',True),
#                                          string=f'MWFvsFit_{noPSOIterPix}Iter',
#                                          valOutliers=valOutliers,cutOutliers=cutOutliers,  
#                                          valPercentile=valPercentile,cutPercentile=cutPercentile)
    
#                 yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
#                 savepath = f'{os.path.dirname(file)}/MWFvsFit_{noPSOIterPix}Iter/'   
#                 np.save(f'{savepath}/y{yy}x{xx}_pixelresult.npy', result_map_px)
#                 PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

#########################################################################################################

# ### (c.1) Post hoc visualization of already calculated MWF maps (slice)

#     if plotResults==True and plotSlice==True:
                
#         for path in resultsDir:
            
#             print(f'\nCurrent directory (PLOT SLICE):\n{path}')
            
#             if len([os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')])==0:
#                 print('ATTENTION: No MWF map files for a slice found in directory! Passed ...')
#                 continue
            
#             file = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')][0]
#             data = np.load(file, allow_pickle=True).item()
            
#             # get parameters from npy and json files
#             PSO.signType  = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
#             PSO.noPeaks   = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
#             signal        = ('').join([sig for sig in PSO.signType])
            
#             if noSlice==18 and useMask==True:
#                 for sig in PSO.signType:
#                     mask          = np.copy(rootMWF.data.msk)
#                     mask[mask==0] = np.nan
#                     data[sig]     = PSO.__cutarray2mask__(data[sig], mask, cut2mask=True)
            
#             with open(os.path.join(path, f'{signal}_param.json'), 'r') as json_file: 
#                 jsonParam     = json.load(json_file) 
            
#             for kk in range(data[PSO.signType[0]].shape[-1]):

#                 if kk<data[PSO.signType[0]].shape[-1]-1:
#                     _PSOgrafics.plotSlice(PSOclass=PSO,PSOresult=data,index=kk,
#                                           saveFig=(f'{path}/',True),limit=(jsonParam,True))
            
#                 if kk==data[PSO.signType[0]].shape[-1]-1:
#                     _PSOgrafics.plotSlice(PSOclass=PSO,PSOresult=data,index=-1,
#                                           saveFig=(f'{path}/',True),limit=(jsonParam,True))
        
#         PSO.log(startTime=startTime, string='\nVisualization slice - Execution time', dim='HMS')

#########################################################################################################
        
# ### (c.2) Post hoc visualization of already calculated MWF maps (pixels)

#     if plotResults==True and plotPixelAcc==True:
        
#         for path in resultsDir:

#             print(f'\nCurrent directory (PLOT PIXEL):\n{path}')
            
#             file = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')][0]
#             data_slice = np.load(file, allow_pickle=True).item()
                    
#             fileList = []
            
#             for dirs, _, files in os.walk(path):
#                 for file in files:
#                     if 'pixelresult' in file: 
#                         fileList.append(os.path.join(dirs,file))
            
#             if len(fileList)==0:
#                 print('ATTENTION: No accuracy files for single pixels found in directory! Passed ...')
#                 continue

#             for file in fileList:
                
#                 data       = np.load(file, allow_pickle=True).item()
#                 filename   = os.path.basename(file)
#                 pattern    = r'(?<!\d)(\d{2,3})(?!\d)'
#                 yy         = int(re.findall(pattern, os.path.basename(file))[0])
#                 xx         = int(re.findall(pattern, os.path.basename(file))[1])
                
#                 # get parameters from npy and json files
#                 PSO.signType  = ['T2'] if 'T2/' in file else (['T2', 'T2S'] if 'T2T2S/' in file else ['T2S'])
#                 PSO.noPeaks   = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
#                 signal        = ('').join([sig for sig in PSO.signType])
                
#                 if noSlice==18 and useMask==True:
#                     for sig in PSO.signType:
#                         mask            = np.copy(rootMWF.data.msk)
#                         mask[mask==0]   = np.nan
#                         data_slice[sig] = PSO.__cutarray2mask__(data_slice[sig], mask, cut2mask=True)
                
#                 with open(os.path.join(path, f'{signal}_param.json'), 'r') as json_file: 
#                     jsonParam     = json.load(json_file)
#                     PSO.noIter    = jsonParam['PSO specifications']['Iterations']
#                     PSO.noPart    = jsonParam['PSO specifications']['Particles']
#                     PSO.noPSOIter = noPSOIterPix

#                 _PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=data,
#                                          PSOresultSLI=data_slice,position=(yy,xx,noSlice), 
#                                          saveFig=(f'{os.path.dirname(file)}/',True),
#                                          valOutliers=valOutliers,cutOutliers=cutOutliers,  
#                                          valPercentile=valPercentile,cutPercentile=cutPercentile)
        
#         PSO.log(startTime=startTime, string='\nVisualization pixel accuracy - Execution time', dim='HMS')