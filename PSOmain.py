# -*- coding: utf-8 -*-
"""
Main script for performing particle swarm optimization (PSO) on MRI invivo data.

@author: Martin Kobe, martin.kobe@ufz.de, martin.kobe@email.de

@status: 03.2024; part of the JIMM Project (DZNE Bonn & UFZ Leipzig)
"""

# import of necessary tools: built-in, installed
import gc, os, re, sys, time, json, concurrent.futures
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


###############################################################################
### User input area: definitions, specifications and initializations ##########
###############################################################################

# data and save path
NII_dataPath  = 'S:/JIMM/MWF_invivo/DZNE_data/nifti'
NII_savePath  = 'S:/JIMM/MWF_invivo/grafics'
NII_keyString = None
# invivo data files
T1_File       = 'NO_FILE'                  # (not measured in the 2023 session)
T2_File       = 'mese.nii'                 # multi echo spin echo 
T2S_File      = 'megre_mean_mag.nii'       # multi echo gradient echo
T2SP_File     = 'megre_mean_mag.nii'       # phase information of T2S
B1_File       = 'dream_corr_resampled.nii' # radiofrequency (RF) transmit field
T2S_TE_File   = 'te.npy'                   # 1D-array of TE times
# additional specifications
dataType      = 'meas'      # 'meas' for invivo data, 'synt' for synthetic data
addNoise      = (False, 50) # tuple: (boolean, relatively added noice 1/SNR)
useMask       = True        # mask valid for slice18, derived from previous JI
pathMask      = f'{NII_dataPath}/maskSlice18.npy'  # location of your mask file
noSlice       = 18          # specify the slice number to be analysed

# PSO algorithm initialization 
noIter        = 100      # number of iteration while executing PSO
noPart        = 400      # number of particles for executing PSO
noPSOIterSli  = 30       # number of cycles of single PSO executions on a slice
noPSOIterPix  = 1000     # number of cycles of single PSO executions on a pixel
# mathematical specifications
lpNorm        = 'L2'     # LP-NORM --> L1-Norm: Manhattan & L2-Norm: Euclidean
invOpt        = 'V0'     # NUMBER of model vectors --> V0: 2 | V1: 1
noPeaks       = 'DIRAC'  # GAUSS: m1 (MW), m2 (FW) | DIRAC: m1 (MW), m2 (AW), m3 (EW)

# mode of PSO application on invivo data
# (booleans) --> choose signal type and inversion type
invT1, invT2, invT2S = False, True, True
singleInv, jointInv  = False, True
# (a) iteration test
noIterTest    = False    # find reasonable iteration numbers (only for single inv)
percentThresh = 0.5      # associated threshold in percent
# (b) parallelized PSO
calcPSO_slice = True    # parallelized PSO over all pixels in a slice
calcPSO_pixel = True    # parallelized PSO over one pixel with noPSOIterPix iterations
pixelList     = [(45,50), (45,55), (37,60), (40,30)] # associated list of pixels
resultsList   = [f'{NII_savePath}/100I400P30PSO/T2/DIRAC/T2_results.npy',
                 f'{NII_savePath}/100I400P30PSO/T2S/DIRAC/T2S_results.npy',
                 f'{NII_savePath}/100I400P25PSO/T2T2S/GAUSS/T2T2S_results.npy',
                 f'{NII_savePath}/100I400P25PSO/T2T2S/DIRAC/T2T2S_results.npy',
                 f'{NII_savePath}/100I400P30PSO_1/T2T2S/DIRAC/T2T2S_results.npy',
                 f'{NII_savePath}/100I400P30PSO_2/T2/DIRAC/T2_results.npy',
                 f'{NII_savePath}/100I400P30PSO_2/T2S/DIRAC/T2S_results.npy',
                 f'{NII_savePath}/100I400P30PSO_2/T2T2S/DIRAC/T2T2S_results.npy',
                 f'{NII_savePath}/100I400P30PSO_3/T2T2S/DIRAC/T2T2S_results.npy',
                 f'{NII_savePath}/100I400P30PSO_4/T2/DIRAC/T2_results.npy',
                 f'{NII_savePath}/100I400P30PSO_4/T2S/DIRAC/T2S_results.npy']      # list with a result file (slice)
# # (c) post hoc visualization
plotResults   = False       # generally plot results from a .npy-file
plotSlice     = True        # visualize MWF map of a slice from PSO results
plotPixelAcc  = True        # visualize accuracy of a pixel from PSO results
valOutliers, cutOutliers = (None, 0.1), True # cut outliers in acc.plot, tuple: (lower,upper)
valPercentile, cutPercentile = (0, 50), True # cut percentile in acc.plot, tuple: (lower,upper)
resultsDir    = [f'{NII_savePath}/100I400P30PSO/T2/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO/T2S/DIRAC/',
                 f'{NII_savePath}/100I400P25PSO/T2T2S/GAUSS/',
                 f'{NII_savePath}/100I400P25PSO/T2T2S/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_1/T2T2S/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_2/T2/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_2/T2S/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_2/T2T2S/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_3/T2T2S/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_4/T2/DIRAC/',
                 f'{NII_savePath}/100I400P30PSO_4/T2S/DIRAC/'] # list with a result dir


###############################################################################
### PSO algorithm preparation area: depends on Users input ####################
###############################################################################

# load data into lists
T1_Files      = hlp.load_data(NII_dataPath,  T1_File,  NII_keyString)    # list
T2_Files      = hlp.load_data(NII_dataPath,  T2_File,  NII_keyString)    # list
T2S_Files     = hlp.load_data(NII_dataPath,  T2S_File, NII_keyString)    # list
# load data into arrays
if len(T1_Files)>1 or len(T2_Files)>1 or len(T2S_Files)>1:
    sys.exit('ATTENTION: Please dont define lists with more than 1 data file for now.')
T2_Data       = hlp.get_measData(T1_Files[0])
T2_Data       = hlp.get_measData(T2_Files[0])
T2S_Data      = hlp.get_measData(T2S_Files[0])

# input parameter array for PSO-class initialization
typeSig       = ['T1','T2']       if invT1 and     invT2  and not invT2S else \
                ['T1','T2S']      if invT1 and     invT2S and not invT2  else \
                ['T2','T2S']      if invT2 and     invT2S and not invT1  else \
                ['T1','T2','T2S'] if invT1 and     invT2  and     invT2S else \
                ['T1']            if invT1 and not invT2  and not invT2S else \
                ['T2']            if invT2 and not invT1  and not invT2S else ['T2S']

signal_input  = [['T1', 'T2', 'T2S', 'dataType', 'noiseBool', 'singleInv', 'jointInv', 'signType'],
                 [invT1, invT2, invT2S, dataType, addNoise, singleInv, jointInv, typeSig]]

# execution exits for wrong paremeter / boolean constellation
if dataType == 'synt':
    sys.exit('ATTENTION: Using dataType==synt is not fully implemented yet.')

if invOpt == 'V1':
    sys.exit('ATTENTION: Using invOpt==V1 is not fully implemented yet.')
    
if lpNorm == 'L1':
    sys.exit('ATTENTION: Using lpNorm==L1 is not fully implemented yet.')
        
if sum(signal_input[1][:3]) == 3:
    sys.exit('ATTENTION: No T1-measurements in current data set available.')
             
if sum(signal_input[1][:3]) >= 2 and singleInv == True:
    sys.exit(f'ATTENTION: To many parameters for a single inversion.\n{sum(signal_input[1][:3])} instead of 1 are given.')
    
if sum(signal_input[1][:3]) == 1 and jointInv == True:
    sys.exit('ATTENTION: Joint inversion needs at minimum 2 different parameters. One is given.')

if singleInv == True and jointInv == True:
    sys.exit('ATTENTION: Cannot perform both single and joint inversion at the same time.')

if singleInv == False and jointInv == False:
    sys.exit('ATTENTION: Didn´t you forget something? Inversion is still set as false.')
    
if jointInv == True and noIterTest == True:
    sys.exit('ATTENTION: Iteration test only possible for a single signal T1, T2, or T2S.')    

# save and load paths for data from recent/previous PSO calculations
if singleInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{str(noPSOIterSli).zfill(2)}PSO/{typeSig[0]}/')
if jointInv==True:
    savepathPSO  = (f'{NII_savePath}/'
                    f'{str(noIter).zfill(3)}I{str(noPart).zfill(3)}P{str(noPSOIterSli).zfill(2)}PSO/{typeSig[0]}{typeSig[1]}/')    

loadpath         = (f'{NII_savePath}/')

if noIterTest==False and calcPSO_slice==True:
    os.makedirs(savepathPSO, exist_ok=True)

# PSO class initialisation to gain access to maiden methods/objects  
PSO     = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                 signal_input=signal_input,noPeaks=noPeaks)

# MWF_analysis class initialisation to gain access to maiden methods/objects
# !!! KW_T2SP file not being used, but if not given as argument, an error occurs: 
            # File ~\Project JIMM\Python\PyJIMM\mwf_t1t2t2s.py:331 in prep_data
            # if (self.slice['T2S'].any()) and (self.slice['CT2S'].any()):
rootMWF = mwf_analysis(data_dir = NII_dataPath,
                       KW_B1    = B1_File,
                       KW_T1    = T1_File,
                       KW_T2    = T2_File,
                       KW_T2S   = T2S_File,
                       KW_T2SP  = T2SP_File,
                       T2S_TE   = T2S_TE_File)

# Calculation of system matrices using tony's mwf_analysis class
PSO.init_MWF_Analysis(rootMWF, noSlice)
PSOgrafics = PSOgrafics()

# Mask invivo data slice to reduce pixels for analysis and save execution time
# this is only valid for slice no18
if noSlice==18 and useMask==True:
    resultsMask = np.load(pathMask, allow_pickle=True).item()
    resultsMask['T2S'][-2,:,:,0][resultsMask['T2S'][-2,:,:,0]>0.125] = np.nan
    rootMWF.data.msk[np.isnan(resultsMask['T2S'][-2,:,:,0])] = 0 # 3581 pixels left

###############################################################################
### PSO parallelization area: functions and methods ###########################
###############################################################################

# (a) loop over different random particle swarm distributions in the model room

def processPixel(args):
    
    yy,xx,rand,modParam,PSO = args
    
    global noPart, noIter, invT1, invT2, invT2S, singleInv, jointInv, typeSig
    
    del PSO; gc.collect()
    
    if modParam[-1]:
        noIter = modParam[-1]['PSO specifications']['Iterations']
        noPart = modParam[-1]['PSO specifications']['Particles']
        invT1  = True if 'T1' in modParam[-1]['PSO specifications']['Signal type'] else False
        invT2  = True if 'T2' in modParam[-1]['PSO specifications']['Signal type'] else False        
        invT2S = True if 'T2S' in modParam[-1]['PSO specifications']['Signal type'] else False        
        singleInv = True if len(modParam[-1]['PSO specifications']['Signal type']) == 1 else False
        jointInv  = True if len(modParam[-1]['PSO specifications']['Signal type']) > 1 else False
        typeSig   = modParam[-1]['PSO specifications']['Signal type']
                                                       
    signal_input  = [['T1', 'T2', 'T2S', 'dataType', 'noiseBool', 'singleInv', 'jointInv', 'signType'],
                     [invT1, invT2, invT2S, dataType, addNoise, singleInv, jointInv, typeSig]]
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterPix,
                                 signal_input=signal_input,randSeed=(rand,True),noPeaks=modParam[1],modParam=modParam)        

    PSO.init_Grid()
    PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))
    
    # obsData from invivo MRI data for a pixel with coordinates [yy,xx,noSlice]
    if singleInv == True:
        filelist            = T2_Files if typeSig == ['T2'] else T2S_Files
        obsData             = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                     pathSignal=filelist[0])
        
    if jointInv == True:
        obsData  = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                          pathT2=T2_Files[0],pathT2S=T2S_Files[0])

    # execute PSO
    PSO.execPSO(obsData=obsData, plotIterTest=False)
    
    # returns dictionary entry for each single pixel coordinate
    if singleInv==True:
        return {f'mod{typeSig[0]}': PSO.globMod[typeSig[0]], 'fit': PSO.globFit[typeSig[0]]}
    
    if jointInv==True:
        signal = ('').join([sig for sig in typeSig])
        return {'modT2':  PSO.globMod[typeSig[0]], 
                'modT2S': PSO.globMod[typeSig[1]], 'fit': PSO.globFit[signal]}

def main_parallel_pixel(yy,xx,modParam):
    
    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                  signal_input=signal_input,noPeaks=noPeaks)
    
    pixelCoord = [(yy,xx,rand,modParam,PSO) for rand in np.random.randint(0, 2**31, size=noPSOIterPix)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # parallelization possible in two consecutive ways
        # a: core based --> executor.map  |  b: thread based on one core --> 
        result_map = list(executor.map(processPixel, pixelCoord))

    return result_map


# (b) loop over all pixels on a given 2D data slice

def processSlice(args):
    
    yy,xx,rand,modParam,PSO = args
    
    del PSO; gc.collect()
    
    if rootMWF.data.msk[yy,xx] == 0:
        
        length = 7 if noPeaks == 'GAUSS' else 10
        
        if singleInv==True:
            return {'pix': [yy,xx], f'mod{typeSig[0]}': np.full(length,np.nan), 'fit': np.nan}

        if jointInv==True:
            signal = ('').join([sig for sig in typeSig])
            return {'pix': [yy,xx], 'modT2': np.full(length,np.nan), 
                    'modT2S': np.full(length,np.nan), 'fit': np.nan}
    
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

    # execute PSO
    PSO.execPSO(obsData=obsData, plotIterTest=False)
    
    # returns dictionary entry fpr each single pixel coordinate
    if singleInv==True:
        return {'pix': [yy,xx], f'mod{typeSig[0]}': PSO.globMod[typeSig[0]], 'fit': PSO.globFit[typeSig[0]]}
    
    if jointInv==True:
        signal = ('').join([sig for sig in typeSig])
        return {'pix': [yy,xx], 'modT2': PSO.globMod[typeSig[0]], 
                'modT2S': PSO.globMod[typeSig[1]], 'fit': PSO.globFit[signal]}

def main_parallel_slice(rand,modParam):

    PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                  signal_input=signal_input,randSeed=(rand, True),noPeaks=noPeaks)
    
    pixel_coordinates = [(yDir, xDir, rand, modParam, PSO) for yDir in range(0, T2_Data.shape[0]) for xDir in range(0,T2_Data.shape[1])]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # parallelization possible in two consecutive ways
        # a: core based --> executor.map  |  b: thread based on one core --> 
        result_map = list(executor.map(processSlice, pixel_coordinates))

    return result_map


###############################################################################
### PSO execution area: different modes possible, depending on Users input ####
###############################################################################

if __name__ == "__main__":

    print(f'Starttime (GMT): {gmtTime}')

###############################################################################       
### (a) Performance test: #####################################################
###     --> how many iterations until diff: fit(n)-fit(n-1) < thresh ##########
    
    if noIterTest == True and singleInv == True:

        sig      = typeSig[0]
        maskList = []
        testList = [(yDir, xDir) for yDir in range(50,58) for xDir in range(45,46)]
    
        for ii, item in enumerate(testList[:]):

            yy, xx = item[0], item[1]
            
            if rootMWF.data.msk[yy,xx] == 0:
                continue
            
            PSO = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,noPSOIter=noPSOIterSli,
                                         signal_input=signal_input,noPeaks=noPeaks)        

            PSO.init_Grid()
            PSO.init_sysMatrix(rootMWF, position=(yy,xx,noSlice))

            filelist     = T2_Files if typeSig == ['T2'] else T2S_Files
            obsData      = PSO.getObsDataFromMeas(position=(yy,xx,noSlice),
                                                  pathSignal=filelist[0])
                
            PSO.execPSO(obsData=obsData,plotIterTest=True)

            aa   = np.abs(np.diff(PSO.globFit_list[sig]) / PSO.globFit_list[sig][:-1] * 100)
            mask = np.abs(aa) >= percentThresh
            maskList.append(np.where(mask)[0][-1])
            
            PSO.log(startTime=startTime, string=f'\nIteration test Pix.[{yy},{xx}]: - Execution time', dim='HMS')
            PSOgrafics.plotIterTest(MWFclass=rootMWF, PSOclass=PSO, position=(yy,xx,noSlice),
                                    savepath=NII_savePath, string=f'PSOIterTest/{typeSig[0]}/{noPeaks}')

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
        
        result_map_sl = {sig: np.zeros([7,  110, 110, noPSOIterSli+1]) if noPeaks == 'GAUSS' 
                         else np.zeros([10, 110, 110, noPSOIterSli+1]) for sig in typeSig}

        signal        = ('').join([sig for sig in typeSig])
        
        for kk, rand in enumerate(np.random.randint(0, 100001, size=noPSOIterSli)):

            print(f'\nExecute PSO {str(kk).zfill(2)} - signal: {signal} | peaks: {noPeaks}')    
    
            results_sl    = main_parallel_slice(rand,modParam=('class', None))
            
            result_map_sl = PSO.result2array(results_sl, result_map_sl, kk,
                                             cutThresh      =(None,None,False), 
                                             cutMask        =(resultsMask,False),
                                             calcBestResult =True,
                                             arrayType      ='Slice')
    
            PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

            PSOgrafics.plotSlice(PSOclass=PSO, PSOresult=result_map_sl, index=kk, 
                                 saveFig=(savepathPSO,True), string=f'{noPeaks}')
            
            if kk==noPSOIterSli-1:
                PSOgrafics.plotSlice(PSOclass=PSO, PSOresult=result_map_sl, index=-1,
                                     saveFig=(savepathPSO,True), string=f'{noPeaks}')
        
        np.save(f'{savepathPSO}{noPeaks}/{signal}_results.npy', result_map_sl)
        PSO.write_JSON(savepath=f'{savepathPSO}{noPeaks}/')

### (b.2) Parallelized calculation of MWF for one pixel with different randoms

    if calcPSO_pixel==True and calcPSO_slice==True:
        
        np.random.seed(0)
             
        result_map_px       = {sig: np.zeros([7,  noPSOIterPix]) if noPeaks == 'GAUSS' 
                               else np.zeros([10, noPSOIterPix]) for sig in typeSig}

        signal              = ('').join([sig for sig in typeSig])        
        
        for kk, (yy,xx) in enumerate(pixelList):

            print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {noPeaks}')      

            results       = main_parallel_pixel(yy,xx,modParam=('class', noPeaks, None))
            
            result_map_px = PSO.result2array(results, result_map_px, kk,
                                             cutThresh      =(None,None,False), 
                                             cutMask        =(resultsMask,False),
                                             calcBestResult =False,
                                             arrayType      ='Pixel')
            
            PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map_px,
                                    PSOresultSLI=result_map_sl,position=(yy,xx,noSlice), 
                                    saveFig=(f'{savepathPSO}{noPeaks}/',True),
                                    string=f'MWFvsFit_{noPSOIterPix}Iter',
                                    valOutliers=(5,0.1),cutOutliers=True,  
                                    valPercentile=(0,50),cutPercentile=True)

            yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
            savepath = f'{savepathPSO}{noPeaks}/MWFvsFit_{noPSOIterPix}Iter/'            
            np.save(f'{savepath}y{yy}x{xx}_pixelresult.npy', result_map_px)
            PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

### (b.3) Parallelized calculation of MWF for one pixel with different randoms

    if calcPSO_pixel==True and calcPSO_slice==False:
        
        for file in resultsList:
            
            print(f'\nCurrent file (PSO CALC PIXEL):\n{file}')
            
            # get parameters from npy and json files
            PSO.signType = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
            PSO.noPeaks  = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            signal       = ('').join([sig for sig in PSO.signType])
            
            result_map_sl = np.load(file, allow_pickle=True).item()
            result_map_px = {sig: np.zeros([7,  noPSOIterPix]) if PSO.noPeaks == 'GAUSS' 
                             else np.zeros([10, noPSOIterPix]) for sig in PSO.signType}
    
            dirPA  = os.path.dirname(file)
            pathPA = os.path.join(dirPA, f'{signal}_param.json')
                                  
            with open(pathPA, 'r') as json_file: 
                jsonParam     = json.load(json_file)
                modParam      = ('json', PSO.noPeaks, jsonParam)
                PSO.noIter    = modParam[-1]['PSO specifications']['Iterations']
                PSO.noPart    = modParam[-1]['PSO specifications']['Particles']
                PSO.noPSOIter = noPSOIterPix
            
            np.random.seed(0)
            
            for kk, (yy,xx) in enumerate(pixelList):

                pixelpath = os.path.join(dirPA, f'MWFvsFIT_{noPSOIterPix}Iter', f'y{yy}x{xx}_pixelresult.npy')

                if os.path.exists(pixelpath):
                    print(f'Already existing: pixel result for y{yy}x{xx}. Continue...')
                    continue
                
                print(f'\nExecute PSO for y{yy}x{xx} - signal: {signal} | peaks: {PSO.noPeaks}')
                
                results       = main_parallel_pixel(yy,xx,modParam=modParam)
                
                result_map_px = PSO.result2array(results, result_map_px, kk,
                                                 cutThresh      =(None,None,False), 
                                                 cutMask        =(resultsMask,False),
                                                 calcBestResult =False,
                                                 arrayType      ='Pixel')        
        
                PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=result_map_px,
                                        PSOresultSLI=result_map_sl,position=(yy,xx,noSlice), 
                                        saveFig=(f'{os.path.dirname(file)}/',True),
                                        string=f'MWFvsFit_{noPSOIterPix}Iter',
                                        valOutliers=valOutliers,cutOutliers=cutOutliers,  
                                        valPercentile=valPercentile,cutPercentile=cutPercentile)
    
                yy, xx   = str(yy).zfill(2), str(xx).zfill(2)
                savepath = f'{os.path.dirname(file)}/MWFvsFit_{noPSOIterPix}Iter/'   
                np.save(f'{savepath}/y{yy}x{xx}_pixelresult.npy', result_map_px)
                PSO.log(startTime=startTime, string='\nExecution time', dim='HMS')

### (c.1) Post hoc visualization of already calculated MWF maps (slice)

    if plotResults==True and plotSlice==True:
                
        for path in resultsDir:
            
            print(f'\nCurrent directory (PLOT SLICE):\n{path}')
            
            if len([os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')])==0:
                print('ATTENTION: No MWF map files for a slice found in directory! Passed ...')
                continue
            
            file = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')][0]
            data = np.load(file, allow_pickle=True).item()
            
            # get parameters from npy and json files
            PSO.signType  = ['T2'] if 'T2_' in file else (['T2', 'T2S'] if 'T2T2S_' in file else ['T2S'])
            PSO.noPeaks   = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
            signal        = ('').join([sig for sig in PSO.signType])
            
            if noSlice==18 and useMask==True:
                for sig in PSO.signType:
                    mask          = np.copy(rootMWF.data.msk)
                    mask[mask==0] = np.nan
                    data[sig]     = PSO.__cutarray2mask__(data[sig], mask, cut2mask=True)
            
            with open(os.path.join(path, f'{signal}_param.json'), 'r') as json_file: 
                jsonParam     = json.load(json_file) 
            
            for kk in range(data[PSO.signType[0]].shape[-1]):

                if kk<data[PSO.signType[0]].shape[-1]-1:
                    PSOgrafics.plotSlice(PSOclass=PSO,PSOresult=data,index=kk,
                                         saveFig=(f'{path}/',True),limit=(jsonParam,True))
            
                if kk==data[PSO.signType[0]].shape[-1]-1:
                    PSOgrafics.plotSlice(PSOclass=PSO,PSOresult=data,index=-1,
                                         saveFig=(f'{path}/',True),limit=(jsonParam,True))
        
        PSO.log(startTime=startTime, string='\nVisualization slice - Execution time', dim='HMS')
        
### (c.2) Post hoc visualization of already calculated MWF maps (pixels)

    if plotResults==True and plotPixelAcc==True:
        
        for path in resultsDir:

            print(f'\nCurrent directory (PLOT PIXEL):\n{path}')
            
            file = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('npy')][0]
            data_slice = np.load(file, allow_pickle=True).item()
                    
            fileList = []
            
            for dirs, _, files in os.walk(path):
                for file in files:
                    if 'pixelresult' in file: 
                        fileList.append(os.path.join(dirs,file))
            
            if len(fileList)==0:
                print('ATTENTION: No accuracy files for single pixels found in directory! Passed ...')
                continue

            for file in fileList:
                
                data       = np.load(file, allow_pickle=True).item()
                filename   = os.path.basename(file)
                pattern    = r'(?<!\d)(\d{2,3})(?!\d)'
                yy         = int(re.findall(pattern, os.path.basename(file))[0])
                xx         = int(re.findall(pattern, os.path.basename(file))[1])
                
                # get parameters from npy and json files
                PSO.signType  = ['T2'] if 'T2/' in file else (['T2', 'T2S'] if 'T2T2S/' in file else ['T2S'])
                PSO.noPeaks   = 'DIRAC' if 'DIRAC' in file else 'GAUSS'
                signal        = ('').join([sig for sig in PSO.signType])
                
                if noSlice==18 and useMask==True:
                    for sig in PSO.signType:
                        mask            = np.copy(rootMWF.data.msk)
                        mask[mask==0]   = np.nan
                        data_slice[sig] = PSO.__cutarray2mask__(data_slice[sig], mask, cut2mask=True)
                
                with open(os.path.join(path, f'{signal}_param.json'), 'r') as json_file: 
                    jsonParam     = json.load(json_file)
                    PSO.noIter    = jsonParam['PSO specifications']['Iterations']
                    PSO.noPart    = jsonParam['PSO specifications']['Particles']
                    PSO.noPSOIter = noPSOIterPix

                PSOgrafics.plotMWFvsFIT(PSOclass=PSO,PSOresultPIX=data,
                                        PSOresultSLI=data_slice,position=(yy,xx,noSlice), 
                                        saveFig=(f'{os.path.dirname(file)}/',True),
                                        valOutliers=valOutliers,cutOutliers=cutOutliers,  
                                        valPercentile=valPercentile,cutPercentile=cutPercentile)
        
        PSO.log(startTime=startTime, string='\nVisualization pixel accuracy - Execution time', dim='HMS')