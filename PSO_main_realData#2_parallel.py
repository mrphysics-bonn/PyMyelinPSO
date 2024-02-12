# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:44:46 2023

@author: Martin Kobe, martin.kobe@ufz.de
"""

# import of necessary tools: built-in, installed
import gc, os, glob, sys, time, json, matplotlib, concurrent.futures
import numpy                      as     np
import pandas                     as     pd
import matplotlib.pyplot          as     plt
from   scipy.optimize             import curve_fit

# import of necessary tools: own scripts and classes
# have to be stored in the same directory as the main script
from   parameters                                 import Parameters as PM
import helpTools                                  as     hlp
from   PSOworkflow_JointInversion_realData_6param import ParticleSwarmOptimizer, PSO_Plots

from   mwf_t1t2t2s                                import mwf_analysis
# try:
#     from   JIMM_repository.mwf_invivo.mwf_t1t2t2s              import mwf_analysis
#     from   JIMM_repository.mwf_invivo.mwf_parametric_inversion import mwfParametricInversion
# except:
#     pass

# time tracking
startTime = time.time()

# parameter initialisation
np.random.seed(0)

noIter    = 100
noPart    = 400
noSlice   = 18
noPSOIter = 30

# data initialisation
# MWF_ is the myelin water fraction of the pixle [xx,yy] in slice [noSclice]
#
# mese:                 multi echo spin echo     --> T2  signal
# megre:                multi echo gradient echo --> T2* signal
# dream_corr_resampled: ???

NII_dataPath  = 'S:/JIMM/real_data/DZNE_data'
NII_savePath  = 'S:/JIMM/real_data/grafics'
NII_keyString = None

###############################################################################
# test PSO algorithm with one file and one slice
T2_Path       = os.path.normpath(os.path.join(NII_dataPath, 'mese'))
T2S_Path      = os.path.normpath(os.path.join(NII_dataPath, 'megre_mean_mag'))

T2_Files      = hlp.load_data(T2_Path,  '.nii', NII_keyString)
T2S_Files     = hlp.load_data(T2S_Path, '.nii', NII_keyString)

T2_Data       = hlp.get_measData(T2_Files[0])
T2S_Data      = hlp.get_measData(T2S_Files[0])

# booleans for the coice of calculated signal type
# for single inversion mark signal of interest as TRUE
# for joint inversion mark signals of interest as TRUE
addNoise      = False
invT1         = False
invT2         = True
invT2S        = True
singleInv     = False
jointInv      = True
lpNorm        = 'L2'            # L1-Norm: Manhattan     | L2-Norm: euklidisch
optionInv     = 'V0'            # V0 - m: 2 , MWF: delta | V1 - m:1 , MWF: same
dataType      = 'meas'
noPeaks       = 'DIRAC'         # GAUSS: m1 (MW), m2 (FW) | DIRAC: m1 (MW), m2 (AW), m3 (EW)

signal_input  = [['T1', 'T2', 'T2S', 'noise', 'singleInv', 'jointInv', 'optionInv', 'dataType'], 
                 [invT1, invT2, invT2S, addNoise, singleInv, jointInv, optionInv, dataType]]

signal_count  = sum(signal_input[1][:3])
array_length  = 4 if jointInv else 3    

if signal_count >= 2 and singleInv == True:
    sys.exit(f'ATTENTION: To many parameters for a single inversion.\n{signal_count} instead of 1 are given.')
    
if signal_count == 1 and jointInv == True:
    sys.exit('ATTENTION: Joint inversion needs at m◙inimum 2 different parameters. One is given.')

if singleInv == True and jointInv == True:
    sys.exit('ATTENTION: Cannot perform both single and joint inversion at the same time.')

if singleInv == False and jointInv == False:
    sys.exit('ATTENTION: Didn´t you forget something? Inversion is still set as false.')    

calcPSO       = True    # PSO with start parameter intervals
calcPSOnarrow = False   # PSO with narrowed parameter intervals after 1st PSO
postProcPSO   = False   # Post processing of PSO result array (cut, ROI, save, ...)
noIterTest    = False   # Find reasonable iteration numbers (only for single inv)
    
###############################################################################
### strings for use in plots of the results and path names for saving results 
typeSig      = ['T1','T2']       if invT1 and     invT2  and not invT2S else \
               ['T1','T2S']      if invT1 and     invT2S and not invT2  else \
               ['T2','T2S']      if invT2 and     invT2S and not invT1  else \
               ['T1','T2','T2S'] if invT1 and     invT2  and     invT2S else \
               ['T1']            if invT1 and not invT2  and not invT2S else \
               ['T2']            if invT2 and not invT1  and not invT2S else ['T2S']

if singleInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{noPSOIter}PSO/{typeSig[0]}/')
if jointInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{noPSOIter}PSO/{typeSig[0]}{typeSig[1]}/')    

loadpath     = (f'{NII_savePath}/')
                # f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{noPSOIter}PSO/')

os.makedirs(savepathPSO, exist_ok=True)

###############################################################################
### result arrays
### can be implemented into PSO class when I have time
### GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
### DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit
if optionInv=='V0':
    result_map = {sig: np.zeros([7, 110, 110, noPSOIter+1]) if noPeaks == 'GAUSS' 
                  else np.zeros([10, 110, 110, noPSOIter+1]) for sig in typeSig}
if optionInv=='V1':
    result_map = {sig: np.zeros([7, 110, 110, noPSOIter+1]) if noPeaks == 'GAUSS' 
                  else np.zeros([10, 110, 110, noPSOIter+1]) for sig in typeSig}    

# PSO class initialisation to gain access to maiden methods/objects  
PSO     = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIter,
                                 signal_input=signal_input, signal_type=typeSig,
                                 randSeed=(0, True),noPeaks=noPeaks)

# MWF_analysis class initialisation to gain access to maiden methods/objects
# !!! KW_T2SP file not being used, but if not given as argument, error occurs: 
            # File ~\Project JIMM\Python\PyJIMM\mwf_t1t2t2s.py:331 in prep_data
            # if (self.slice['T2S'].any()) and (self.slice['CT2S'].any()):
rootMWF = mwf_analysis(data_dir = os.path.join(NII_dataPath, 'nifti'),
                       KW_B1    = 'dream_corr_resampled.nii',
                       KW_T2    = 'mese.nii',
                       KW_T2S   = 'megre_mean_mag.nii',
                       KW_T2SP  = 'megre_mean_mag.nii',
                       T2S_TE   = 'te.npy')

# Calculation of system matrices using tony's mwf_analysis class
PSO.init_MWF_Analysis(rootMWF, noSlice)

# Initalization of the grafics class to plot PSO results in various ways
PSO_plot = PSO_Plots()

# # Idee: tools for writing, re-arranging data, open files, etc...
# PSO_tools = 
    
###############################################################################
###############################################################################
### Parallelization: loop over pixels

def process_pixel(args):
    
    yy, xx, PSO, rand = args
    
    if rootMWF.data.msk[yy,xx] == 0:
        
        length = 7 if noPeaks == 'GAUSS' else 10
        
        if singleInv==True:
            return {'pix': [yy,xx], f'mod{typeSig[0]}': np.full(length,np.nan), 'fit': np.nan}

        if jointInv==True:
            signal = ('').join([sig for sig in typeSig])
            return {'pix': [yy,xx], 'modT2': np.full(length,np.nan), 
                    'modT2S': np.full(length,np.nan), 'fit': np.nan}
    
    del PSO
    gc.collect()
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIter,signal_input=signal_input,
                                 signal_type=typeSig,randSeed=(rand, True),noPeaks=noPeaks)        
    PSO.init_Grid()
    PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
    
    ### obsData from measurement for a pixel with coordinates [yy,xx,noSlice]
    if singleInv == True:
        filelist            = T2_Files if typeSig == ['T2'] else T2S_Files
        obsData             = PSO.getObsDataFromMeas(pathSignal=filelist[0],position=(yy,xx,noSlice))
        
    if jointInv == True:
        obsData  = PSO.getObsDataFromMeas(pathT1=None,
                                          pathT2=T2_Files[0], 
                                          pathT2S=T2S_Files[0], 
                                          position=(yy,xx,noSlice))

    ### creating data for each PSO cycle for further use (writing, plotting, ...)
    #   possible kwargs: matrix_version
    #   Memo an mich: hier müssen noch entsprechend Parameter als kwargs gesetzt werden
    PSO.execPSO(obsData=obsData, matrix_version='V2',
                indPix=None, PSOiter=None, savepath=None)
    
    # returns dictionary entry fpr each single pixel coordinate
    if singleInv==True:
        return {'pix': [yy,xx], f'mod{typeSig[0]}': PSO.globMod[typeSig[0]], 'fit': PSO.globFit[typeSig[0]]}
    
    if jointInv==True:
        signal = ('').join([sig for sig in typeSig])
        return {'pix': [yy,xx], 'modT2': PSO.globMod[typeSig[0]], 
                'modT2S': PSO.globMod[typeSig[1]], 'fit': PSO.globFit[signal]}

def main_parallel(rand):
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIter,signal_input=signal_input,
                                 signal_type=typeSig,randSeed=(rand, True),noPeaks=noPeaks)
    
    pixel_coordinates = [(yDir, xDir, PSO, rand) for yDir in range(0,110) for xDir in range(0,110)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # parallelization possible in two consecutive ways
        # a: core based --> executor.map  |  b: thread based on one core --> 
        result_map = list(executor.map(process_pixel, pixel_coordinates))

    return result_map

###############################################################################     
###############################################################################
### here starts the PSO algorithm using a self written PSO class ##############
### --> (translated from a MATLAB version of a lecture at Uni Potsdam - H.P.) #

if __name__ == "__main__":
        
    # # PSO class initialisation to gain access to maiden methods/objects  
    # PSO     = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIter,
    #                                  signal_input=signal_input, signal_type=typeSig,
    #                                  randSeed=(0, True),noPeaks=noPeaks)
    
    # # MWF_analysis class initialisation to gain access to maiden methods/objects
    # # !!! KW_T2SP file not being used, but if not given as argument, error occurs: 
    #             # File ~\Project JIMM\Python\PyJIMM\mwf_t1t2t2s.py:331 in prep_data
    #             # if (self.slice['T2S'].any()) and (self.slice['CT2S'].any()):
    # rootMWF = mwf_analysis(data_dir = os.path.join(NII_dataPath, 'nifti'),
    #                        KW_B1    = 'dream_corr_resampled.nii',
    #                        KW_T2    = 'mese.nii',
    #                        KW_T2S   = 'megre_mean_mag.nii',
    #                        KW_T2SP  = 'megre_mean_mag.nii',
    #                        T2S_TE   = 'te.npy')

    # # Calculation of system matrices using tony's mwf_analysis class
    # PSO.init_MWF_Analysis(rootMWF, noSlice)
    
    # # Initalization of the grafics class to plot PSO results in various ways
    # PSO_plot = PSO_Plots()
    
    # # # Idee: tools for writing, re-arranging data, open files, etc...
    # # PSO_tools = 

###############################################################################     
### (a) Performance test: how many iterations until diff: fit(n)-fit(n-1) < 0.5
    
    if noIterTest == True:

        sig      = typeSig[0]
        maskList = []
        testList = [(yDir, xDir) for yDir in range(45,46) for xDir in range(55,56)]
    
        for ii, item in enumerate(testList[:]):
         
            yy,xx = item[0],item[1]

            if rootMWF.data.msk[yy,xx] == 0:
                continue
                
            del PSO
            gc.collect()
            
            PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIter,signal_input=signal_input,
                                         signal_type=typeSig,randSeed=(0, True),noPeaks=noPeaks)        

            PSO.init_Grid()
            PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
            
            if singleInv == True:
                filelist     = T2_Files if typeSig == ['T2'] else T2S_Files
                obsData      = PSO.getObsDataFromMeas(pathSignal=filelist[0],position=(yy,xx,noSlice))
                # obsData[sig] = obsData[sig]/np.max(obsData[sig])

            if jointInv == True:
                obsData  = PSO.getObsDataFromMeas(pathT1=None,
                                                  pathT2=T2_Files[0], 
                                                  pathT2S=T2S_Files[0], 
                                                  position=(yy,xx,noSlice))
                
            PSO.execPSO(obsData=obsData, matrix_version='V2', 
                        indPix=None, PSOiter=None, savepath=None, plotResult=True)

            aa   = np.abs(np.diff(PSO.globFit_list[sig]) / PSO.globFit_list[sig][:-1] * 100)
            mask = np.abs(aa) >= 0.5
            maskList.append(np.where(mask)[0][-1])
            PSO.log(startTime=startTime, string=f'\nID {ii} - Execution time', dim='HMS')
            PSO_plot.plotPixel(MWFclass=rootMWF, PSOclass=PSO, position=(yy,xx,noSlice),
                               savepath=savepathPSO, string=f'PSOSingle/{noPeaks}')

        sys.exit()

###############################################################################        
### (b.1) Parallelized calculation of MWF for one slice and with n PSO cycles   
###       Result_map GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
###       Result_map DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit
    if calcPSO==True:
        
        # resMask is a good solution for slice 18 --> probably not usable for other slices 
        resultsMask = np.load('S:/JIMM/real_data/grafics/mask.npy')
        coords      = [(yDir, xDir) for yDir in range(0,110) for xDir in range(0,110)]
        signal      = ('').join([sig for sig in typeSig])
        
        for kk, rand in enumerate(np.random.randint(0, 100001, size=noPSOIter)):
    
            print(f'\nExecute PSO - signal: {signal} | peaks: {noPeaks} | PSO_iter: {kk}')
    
            results    = main_parallel(rand)
            
            result_map = PSO.result2array(results, result_map, kk,
                                          cutThresh      =(None,None,False), 
                                          cutMask        =(resultsMask,True),
                                          calcBestResult =True)
    
            PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')
            PSO_plot.plotSliceJI(PSOclass=PSO, PSOresult=result_map, index=kk, 
                                 saveFig=(savepathPSO,True), string=f'lvl1/{noPeaks}')
        
        np.save(f'{savepathPSO}lvl1/{noPeaks}/{signal}_results.npy', result_map)
        PSO.write_JSON(savepath=savepathPSO)
        
###############################################################################        
### (b.2) Load and postprocess calculated PSO data: calc best fit and cut 2 mask/thresh
###       Result_map GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
###       Result_map DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit
    if postProcPSO==True:
        
        resultsMask = np.load('S:/JIMM/real_data/grafics/mask.npy')
        resultsList = hlp.load_PSO_data(loadpath, 'results.npy')
        
        for file in resultsList:
            
            # get data from file
            PSO.signType = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
            PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            
            if PSO.noPeaks == 'GAUSS': continue

            try:
                result_map = np.load(file, allow_pickle=True).item()
            except:
                result_map                  = {sig:[] for sig in PSO.signType}
                result_map[PSO.signType[0]] = np.load(file)

            coords = [(yDir, xDir) for yDir in range(0,110) for xDir in range(0,110)]
            signal = ('').join([sig for sig in PSO.signType])
            
            # json-file parameters
            dirPA  = os.path.dirname(file)
            pathPA = os.path.join(dirPA, f'{signal}_param.json')
                                  
            with open(pathPA, 'r') as json_file: 
                jsonParam = json.load(json_file)
            
            # calculate best fit values and store in result array [:,:,:,-1]
            for sig in PSO.signType:
                for position in coords:
                    result_map[sig] = PSO.__bestfit2array__(result_map[sig],  position)
                    resultsMask     = PSO.__bestfit2array__(resultsMask, position)
                
                # save results map to array including best fit values in position [:,:,:,-1]
                np.save(f'{os.path.dirname(file)}\\{signal}_results_raw.npy', result_map)
            
                # cut to a mask for presentation (valid only for slice 18)
                result_map[sig] = PSO.__cutarray2mask__(result_map[sig], resultsMask, cut2mask=True)
                
                # # cut to a threshold in MWF or fit: thershold=(MWF, fit)
                # result_map[sig] = PSO.__cutarray2thresh__(result_map[sig], threshold=(0.25,0.004), cut2thresh=False)
                
                # # cut final plot to a ROI
                # null_columns = np.all(rootMWF.data.msk[0:110, 0:110] == 0, axis=0)
                # null_lines   = np.all(rootMWF.data.msk[0:110, 0:110] == 0, axis=1)
                
                # null_ind_col = np.where(np.diff(null_columns))[0]
                # null_ind_lin = np.where(np.diff(null_lines))[0]
                
                # result_map[sig] = np.copy(result_map[sig][:,null_ind_lin[0]-2:null_ind_lin[1]+2, 
                #                                           null_ind_col[0]-2:null_ind_col[1]+2,:])
            
                # # save cutted plots
                # np.save(f'{os.path.dirname(file)}\\{signal}_results_cut.npy', result_map)
            
            # plot bestfit/bestMWF slice content
            for kk in range(result_map[sig].shape[-1]):
                if kk!=PSO.noPSOIter: continue
                PSO_plot.plotSliceJI(PSOclass=PSO, PSOresult=result_map, index=-1,
                                     saveFig=(f'{os.path.dirname(file)}\\',True), limit=(jsonParam,True))
                
            PSO.log(startTime=startTime, string='\npostProcPSO - Execution time', dim='HMS')

###############################################################################        
### (c) Find the best group of values and narrow integration parameters 
###     Result_map GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
###     Result_map DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit 
    if calcPSOnarrow==True:

        resultsList  = hlp.load_PSO_data(loadpath, 'results_cut.npy')
        resultsList2 = hlp.load_PSO_data(loadpath, 'results_raw.npy')
        
        for file in resultsList2:
            
            # plot MWF versus best global fit for a pixel on uncut data (for position)
            PSO.signType = ['T2'] if 'T2_' in file else ['T2S']
            PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            result_map_raw = np.load(file)
            PSO_plot.plotMWFvsFIT(PSOclass=PSO, PSOresult=result_map_raw, position=(45,55),
                                  saveFig=(f'{os.path.dirname(file)}\\', True), string='MWFvsFIT',
                                  valOutliers=(5,0.1), cutOutliers=True,  valPercentile=(0,95), cutPercentile=True)
        
        for file in resultsList:
            
            # get data from file
            PSO.signType = ['T2'] if 'T2_' in file else ['T2S']
            PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            result_map   = np.load(file)
            yy,xx        = result_map.shape[1], result_map.shape[2]
            coords       = [(yDir, xDir) for yDir in range(0,yy) for xDir in range(0,xx)]
            numberBins   = 20
            
            for yy,xx in coords[3500:3530]:
                
                if np.isnan(result_map[-1, yy, xx, :]).any(): continue
    
                hist, bins = np.histogram(result_map[-1, yy, xx, :-1], bins=numberBins, density=False)
                res2       = result_map[:,yy,xx,result_map[-1,yy,xx,:]<=bins[1]]
                
                # PSO_plot.plotHist(PSOclass=PSO, PSOresult=result_map, PSOresultCUT=res2, position=(yy,xx), 
                #                   noBins=numberBins, saveFig=(f'{os.path.dirname(file)}\\', True), string='HIST')
                
                PSO_plot.plotMWFvsFIT(PSOclass=PSO, PSOresult=result_map, position=(yy,xx),
                                      saveFig=(f'{os.path.dirname(file)}\\', True), string='MWFvsFIT',
                                      valOutliers=(5,0.1), cutOutliers=True,  valPercentile=(0,95), cutPercentile=True)