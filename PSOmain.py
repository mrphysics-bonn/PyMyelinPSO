# -*- coding: utf-8 -*-
"""
Main script for performing particle swarm optimization (PSO) on MRI invivo data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)
"""

# import of necessary tools: built-in, installed
import gc, os, sys, time, json, copy, concurrent.futures
import numpy as np

# import of necessary tools: own scripts and classes
# have to be stored in the same directory as the main script
import helpTools   as     hlp
from   PSOworkflow import ParticleSwarmOptimizer
from   PSOplots    import PSOgrafics
from   mwf_t1t2t2s import mwf_analysis

# time tracking
startTime = time.time()
gmtTime   = time.strftime('%H:%M:%S', time.gmtime())

# random necessary for comparability of PSO runs with different parameters
np.random.seed(0)

# parameter initialisation 
noIter       = 100     # number of iteration while executing PSO
noPart       = 400     # number of particles for executing PSO
noPSOIterSli = 25      # number of cycles of single PSO executions on a slice
noPSOIterPix = 1000    # number of cycles of single PSO executions on a pixel 

# data initialisation
# mese:  multi echo spin echo     --> T2  signal
# megre: multi echo gradient echo --> T2* signal
# dream_corr_resampled: multi echo phase information 
NII_dataPath  = 'S:/JIMM/invivoData/DZNE_data'
NII_savePath  = 'S:/JIMM/invivoData/grafics'
NII_keyString = None
noSlice       = 18

T2_Path       = os.path.normpath(os.path.join(NII_dataPath, 'mese'))
T2S_Path      = os.path.normpath(os.path.join(NII_dataPath, 'megre_mean_mag'))

T2_Files      = hlp.load_data(T2_Path,  '.nii', NII_keyString)
T2S_Files     = hlp.load_data(T2S_Path, '.nii', NII_keyString)

T2_Data       = hlp.get_measData(T2_Files[0])
T2S_Data      = hlp.get_measData(T2S_Files[0])


# booleans for the coice of calculated signal type
invT1, invT2, invT2S = False, True, True
singleInv, jointInv  = False, True


# specifications
addNoise      = False    # parameter left from algorithm performance tests
dataType      = 'meas'   # 'meas' for invivo data
lpNorm        = 'L2'     # LP-NORM --> L1-Norm: Manhattan & L2-Norm: euklidisch
invOpt        = 'V0'     # METHOD for inversion --> V0: 2 modvec & MWF| V1: 1 modvec & same MWF
noPeaks       = 'GAUSS'  # GAUSS: m1 (MW), m2 (FW) | DIRAC: m1 (MW), m2 (AW), m3 (EW)

calcPSO_slice = False    # parallelized PSO over all pixels in a slice
calcPSO_pixel = True     # parallelized PSO over one pixel with different rand 
calcPSOnarrow = False    # calc PSO with narrowed parameter intervals after 1st PSO
postProcPSO   = False    # Post processing of PSO result array (cut, ROI, save, ...)
noIterTest    = False    # Find reasonable iteration numbers (only for single inv)


# input parameter array for PSO-class initiation
typeSig       = ['T1','T2']       if invT1 and     invT2  and not invT2S else \
                ['T1','T2S']      if invT1 and     invT2S and not invT2  else \
                ['T2','T2S']      if invT2 and     invT2S and not invT1  else \
                ['T1','T2','T2S'] if invT1 and     invT2  and     invT2S else \
                ['T1']            if invT1 and not invT2  and not invT2S else \
                ['T2']            if invT2 and not invT1  and not invT2S else ['T2S']

signal_input  = [['T1', 'T2', 'T2S', 'dataType', 'noiseBool', 'singleInv', 'jointInv', 'signType'],
                 [invT1, invT2, invT2S, dataType, addNoise, singleInv, jointInv, typeSig]]


# execution exits for wrong paremeter / boolean constellation
if sum(signal_input[1][:3]) == 3:
    sys.exit('ATTENTION: No T1-measurements in current data set available.')
             
if sum(signal_input[1][:3]) >= 2 and singleInv == True:
    sys.exit(f'ATTENTION: To many parameters for a single inversion.\n{sum(signal_input[1][:3])} instead of 1 are given.')
    
if sum(signal_input[1][:3]) == 1 and jointInv == True:
    sys.exit('ATTENTION: Joint inversion needs at m◙inimum 2 different parameters. One is given.')

if singleInv == True and jointInv == True:
    sys.exit('ATTENTION: Cannot perform both single and joint inversion at the same time.')

if singleInv == False and jointInv == False:
    sys.exit('ATTENTION: Didn´t you forget something? Inversion is still set as false.')
    
if jointInv == True and noIterTest == True:
    sys.exit('ATTENTION: Iteration test only possible for a single signal T1, T2, or T2S.')
    

# save and load paths for data from recent/previous PSO calculations
if singleInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{noPSOIterSli}PSO/{typeSig[0]}/')
if jointInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{noPSOIterSli}PSO/{typeSig[0]}{typeSig[1]}/')    

loadpath         = (f'{NII_savePath}/')  # used if postProcPSO = True

os.makedirs(savepathPSO, exist_ok=True)

# PSO class initialisation to gain access to maiden methods/objects  
PSO     = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                 signal_input=signal_input,noPeaks=noPeaks)

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
PSOgrafics = PSOgrafics()

# Mask invivo data slice to reduce pixels for analysis and save execution time
resultsMask = np.load('S:/JIMM/invivoData/grafics/mask2.npy', allow_pickle=True).item()
resultsMask['T2S'][-2,:,:,0][resultsMask['T2S'][-2,:,:,0]>0.125] = np.nan

rootMWF.data.msk[np.isnan(resultsMask['T2S'][-2,:,:,0])] = 0 # 3581 pixels left

# coords      = [(yDir, xDir) for yDir in range(0,110) for xDir in range(0,110)]
# testMask    = np.copy(rootMWF.data.msk)

# for y,x in coords:
#     if rootMWF.data.msk[y,x]==0: continue
#     else:
#         if rootMWF.data.msk[y+1,x]==0 or rootMWF.data.msk[y-1,x]==0 or rootMWF.data.msk[y,x+1]==0 or rootMWF.data.msk[y,x-1]==0:
#             testMask[y,x] = 2

###############################################################################
###############################################################################
### Parallelization: loop over pixels

def processPixel(args):
    
    yy,xx,rand,modParam = args
    
    # del PSO; gc.collect()
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterPix,
                                 signal_input=signal_input,randSeed=(rand,True),noPeaks=noPeaks,modParam=modParam)        
    PSO.init_Grid()
    PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
    
    ### obsData from measurement for a pixel with coordinates [yy,xx,noSlice]
    if singleInv == True:
        filelist            = T2_Files if typeSig == ['T2'] else T2S_Files
        obsData             = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                     pathSignal=filelist[0])
        
    if jointInv == True:
        obsData  = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                          pathT2=T2_Files[0],pathT2S=T2S_Files[0])

    # creating data for each PSO cycle for further use (writing, plotting, ...)
    PSO.execPSO(obsData=obsData, plotIterTest=False)

    # returns dictionary entry for each single pixel coordinate
    if singleInv==True:
        return {f'mod{typeSig[0]}': PSO.globMod[typeSig[0]], 'fit': PSO.globFit[typeSig[0]]}
    
    if jointInv==True:
        signal = ('').join([sig for sig in typeSig])
        return {'modT2':  PSO.globMod[typeSig[0]], 
                'modT2S': PSO.globMod[typeSig[1]], 'fit': PSO.globFit[signal]}

def main_parallel_pixel(yy,xx,modParam):
    
    pixelCoord = [(yy,xx,rand,modParam) for rand in np.random.randint(0, 2**31, size=noPSOIterPix)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # parallelization possible in two consecutive ways
        # a: core based --> executor.map  |  b: thread based on one core --> 
        result_map = list(executor.map(processPixel, pixelCoord))

    return result_map


###############################################################################
###############################################################################
### Parallelization: loop over pixels

def processSlice(args):
    
    yy,xx,rand,modParam = args
    
    if rootMWF.data.msk[yy,xx] == 0:
        
        length = 7 if noPeaks == 'GAUSS' else 10
        
        if singleInv==True:
            return {'pix': [yy,xx], f'mod{typeSig[0]}': np.full(length,np.nan), 'fit': np.nan}

        if jointInv==True:
            signal = ('').join([sig for sig in typeSig])
            return {'pix': [yy,xx], 'modT2': np.full(length,np.nan), 
                    'modT2S': np.full(length,np.nan), 'fit': np.nan}
    # try:
    #     del PSO; gc.collect()
    # except:
    #     print('could not delete PSO instance')
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                 signal_input=signal_input,randSeed=(rand,True),noPeaks=noPeaks,modParam=modParam)        
    PSO.init_Grid()
    PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
    
    ### obsData from measurement for a pixel with coordinates [yy,xx,noSlice]
    if singleInv == True:
        filelist            = T2_Files if typeSig == ['T2'] else T2S_Files
        obsData             = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                     pathSignal=filelist[0])
        
    if jointInv == True:
        obsData  = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                          pathT2=T2_Files[0],pathT2S=T2S_Files[0])

    # creating data for each PSO cycle for further use (writing, plotting, ...)
    PSO.execPSO(obsData=obsData, plotIterTest=False)
    
    # returns dictionary entry fpr each single pixel coordinate
    if singleInv==True:
        return {'pix': [yy,xx], f'mod{typeSig[0]}': PSO.globMod[typeSig[0]], 'fit': PSO.globFit[typeSig[0]]}
    
    if jointInv==True:
        signal = ('').join([sig for sig in typeSig])
        return {'pix': [yy,xx], 'modT2': PSO.globMod[typeSig[0]], 
                'modT2S': PSO.globMod[typeSig[1]], 'fit': PSO.globFit[signal]}

def main_parallel_slice(rand,modParam):
    
    # PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
    #                               signal_input=signal_input,randSeed=(rand, True),noPeaks=noPeaks)
    
    pixel_coordinates = [(yDir, xDir, rand, modParam) for yDir in range(40,50) for xDir in range(45,55)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # parallelization possible in two consecutive ways
        # a: core based --> executor.map  |  b: thread based on one core --> 
        result_map = list(executor.map(processSlice, pixel_coordinates))

    return result_map


###############################################################################     
###############################################################################
### here starts the PSO algorithm using a self written PSO class ##############
### --> (translated from a MATLAB version of a lecture at Uni Potsdam - H.P.) #

if __name__ == "__main__":

    print(f'Starttime (GMT): {gmtTime}')
    
###############################################################################     
### (a) Performance test: #####################################################
###     --> how many iterations until diff: fit(n)-fit(n-1) < thresh ##########
    
    if noIterTest == True and singleInv == True:

        sig      = typeSig[0]
        maskList = []
        thresh   = 0.5                                      # dimension percent
        testList = [(yDir, xDir) for yDir in range(50,58) for xDir in range(45,46)]
    
        for ii, item in enumerate(testList[:]):

            yy, xx = item[0], item[1]
            
            if rootMWF.data.msk[yy,xx] == 0:
                continue
            
            PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                         signal_input=signal_input,noPeaks=noPeaks)        

            PSO.init_Grid()
            PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
            
            if singleInv == True:
                filelist     = T2_Files if typeSig == ['T2'] else T2S_Files
                obsData      = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                      pathSignal=filelist[0])
                
            if jointInv == True:
                obsData  = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                  pathT2=T2_Files[0], pathT2S=T2S_Files[0])
                
            PSO.execPSO(obsData=obsData,plotIterTest=True)

            aa   = np.abs(np.diff(PSO.globFit_list[sig]) / PSO.globFit_list[sig][:-1] * 100)
            mask = np.abs(aa) >= thresh
            maskList.append(np.where(mask)[0][-1])
            
            PSO.log(startTime=startTime, string=f'\nIteration test Pix.[{yy},{xx}]: - Execution time', dim='HMS')
            PSOgrafics.plotPixel(MWFclass=rootMWF, PSOclass=PSO, position=(yy,xx,noSlice),
                                 savepath=savepathPSO, string=f'PSOSingle/{noPeaks}')

            print(f'Number of iterations before threshold is reached: {np.where(mask)[0][-1]}\n')

            del PSO; gc.collect()
                        
        sys.exit(f'Iteration test finished. Worst mask ID {np.max(maskList)}')

###############################################################################        
### (b.1) Parallelized calculation of MWF for one slice and with n PSO cycles
###       invOpt V0 --> Each signal has a respective model vector. 
###                     MWF is present in the objective function.
###      
###       invOpt V1 --> One model vector for both signals combined (same MWF).
###                     MWF is not present in the objective function.
###
### Result_map GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
### Result_map DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit

    if calcPSO_slice==True:        
        
        result_map = {sig: np.zeros([7,  110, 110, noPSOIterSli+1]) if noPeaks == 'GAUSS' 
                      else np.zeros([10, 110, 110, noPSOIterSli+1]) for sig in typeSig}

        signal      = ('').join([sig for sig in typeSig])
        
        for kk, rand in enumerate(np.random.randint(0, 100001, size=noPSOIterSli)):

            print(f'\nExecute PSO {str(kk).zfill(2)} - signal: {signal} | peaks: {noPeaks}')    
    
            results    = main_parallel_slice(rand,modParam=('class', None))
            
            result_map = PSO.result2array(results, result_map, kk,
                                          cutThresh      =(None,None,False), 
                                          cutMask        =(resultsMask,False),
                                          calcBestResult =True,
                                          arrayType      ='Slice')
    
            PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

            PSOgrafics.plotSlice(PSOclass=PSO, PSOresult=result_map, index=kk, 
                                 saveFig=(savepathPSO,True), string=f'{noPeaks}')
        
        np.save(f'{savepathPSO}{noPeaks}/{signal}_results.npy', result_map)
        PSO.write_JSON(savepath=f'{savepathPSO}{noPeaks}/')

### (b.2) Parallelized calculation of MWF for one pixel with different randoms

    if calcPSO_pixel==True and calcPSO_slice==True:
        
        np.random.seed(0)
        
        PSO.noPSOIter    = noPSOIterPix
        coordList        = [(45,50), (45,55), (37,60), (40,30)]
        
        result_map_slice = copy.deepcopy(result_map)        
        result_map       = {sig: np.zeros([7,  noPSOIterPix]) if noPeaks == 'GAUSS' 
                            else np.zeros([10, noPSOIterPix]) for sig in typeSig}

        signal           = ('').join([sig for sig in typeSig])        
        
        for kk, (yy,xx) in enumerate(coordList[:1]):

            print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {noPeaks}')      
            
            results    = main_parallel_pixel(yy,xx,modParam=('class', None))
            
            result_map = PSO.result2array(results, result_map, kk,
                                          cutThresh      =(None,None,False), 
                                          cutMask        =(resultsMask,False),
                                          calcBestResult =False,
                                          arrayType      ='Pixel')
            
            PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map,
                                    PSOresultSLI=result_map_slice,position=(yy,xx,noSlice), 
                                    saveFig=(f'{savepathPSO}{noPeaks}/',True),
                                    string=f'MWFvsFit_{noPSOIterPix}Iter',
                                    valOutliers=(5,0.1),cutOutliers=True,  
                                    valPercentile=(0,50),cutPercentile=True)

            savepath   = f'{savepathPSO}{noPeaks}/MWFvsFit_{noPSOIterPix}Iter/'            
            np.save(f'{savepath}y{yy}_x{xx}_pixelresult.npy', result_map)
            PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

        PSO.write_JSON(savepath=f'{savepath}/')

### (b.3) Parallelized calculation of MWF for one pixel with different randoms

    if calcPSO_pixel==True and calcPSO_slice==False:
        
        coordList   = [(45,50), (45,55), (37,60), (40,30)]
        resultsList = hlp.load_PSO_data(loadpath, 'results.npy')
        
        # resultsList = ['path of *_results.npy - file 1', 
        #                'path of *_results.npy - file 2',
        #                'path of *_results.npy - file 3', ...]
        
        for file in resultsList[:1]:

            result_map_slice = np.load(file, allow_pickle=True).item()
            result_map       = {sig: np.zeros([7,  noPSOIterPix]) if noPeaks == 'GAUSS' 
                                else np.zeros([10, noPSOIterPix]) for sig in typeSig}        
            
            # get parameters from npy and json files
            PSO.signType  = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
            PSO.noPeaks   = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            PSO.noPSOIter = noPSOIterPix
            signal        = ('').join([sig for sig in PSO.signType])

            dirPA  = os.path.dirname(file)
            pathPA = os.path.join(dirPA, f'{signal}_param.json')
                                  
            with open(pathPA, 'r') as json_file: 
                jsonParam     = json.load(json_file)
                modParam      = ('json', jsonParam)
            
            np.random.seed(0)
            
            for kk, (yy,xx) in enumerate(coordList[-1:]):
    
                print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {noPeaks}')      
                
                results    = main_parallel_pixel(yy,xx,modParam=modParam)
                
                result_map = PSO.result2array(results, result_map, kk,
                                              cutThresh      =(None,None,False), 
                                              cutMask        =(resultsMask,False),
                                              calcBestResult =False,
                                              arrayType      ='Pixel')        
        
                PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map,
                                        PSOresultSLI=result_map_slice,position=(yy,xx,noSlice), 
                                        saveFig=(f'{os.path.dirname(file)}/',True),
                                        string=f'MWFvsFit_{noPSOIterPix}Iter',
                                        valOutliers=(5,0.1),cutOutliers=True,  
                                        valPercentile=(0,50),cutPercentile=True)

                np.save(f'{os.path.dirname(file)}/MWFvsFit_{noPSOIterPix}Iter/y{yy}_x{xx}_pixelresult.npy', result_map)
                PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')


###############################################################################        
### (c.1) Load and postprocess calculated PSO data: calc best fit and cut 2 mask/thresh
###       Result_map GAUSS-0:m1,1:m1_sig,2:m2,3:m2_sig,4:int2,5:MWF,6:misfit
###       Result_map DIRAC-0:m1,1:m1_sig,2:m2,3:m2_sig,4:m3,5:m3_sig,6:int2,7:int3,8:MWF,9:misfit

    if postProcPSO==True:
        
        resultsMask = np.load('S:/JIMM/real_data/grafics/mask2.npy', allow_pickle=True).item()
        resultsList = hlp.load_PSO_data(loadpath, 'results.npy')
        
        for file in resultsList:
            
            # get data from file
            PSO.signType = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
            PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'

            result_map = np.load(file, allow_pickle=True).item()

            coords = [(yDir, xDir) for yDir in range(45,46) for xDir in range(45,46)]
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
            
                # cut to a mask for presentation (valid only for slice 18)
                result_map[sig] = PSO.__cutarray2mask__(result_map[sig], 
                                                        resultsMask, cut2mask=False)
                
                # cut to a threshold in MWF or fit: threshold=(MWF, fit)
                result_map[sig] = PSO.__cutarray2thresh__(result_map[sig], 
                                                          threshold=(0.25,0.004), cut2thresh=False)
                
            # save results map to array including best fit values in position [:,:,:,-1]
            np.save(f'{os.path.dirname(file)}\\{signal}_results_maskcut.npy', result_map)
                
            # plot bestfit/bestMWF slice content
            for kk in range(result_map[sig].shape[-1]):
                if kk!=PSO.noPSOIter: continue
                PSOgrafics.plotSlice(PSOclass=PSO, PSOresult=result_map, index=-1,
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
            PSOgrafics.plotMWFvsFIT(PSOclass=PSO, PSOresult=result_map_raw, position=(45,55),
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
                
                PSOgrafics.plotMWFvsFIT(PSOclass=PSO, PSOresult=result_map, position=(yy,xx),
                                        saveFig=(f'{os.path.dirname(file)}\\', True), string='MWFvsFIT',
                                        valOutliers=(5,0.1), cutOutliers=True,  valPercentile=(0,95), cutPercentile=True)