# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:24:44 2023

@author: kobe
"""

''' test the forward solvers '''


import gc, os, sys, time, json, warnings, psutil
import numpy                      as np
import pandas                     as pd
import matplotlib.pyplot          as plt
from   sklearn.cluster            import KMeans

# import of necessary tools: own scripts and classes
# have to be stored in the same directory as the main script
import helpTools                  as     hlp
from   helpToolsT2                import EpyG_Paramset, EpyG_mSE_system_matrix
from   kernels                    import T1_decay_abs, T2_decay, system_matrix_from_kernel
from   parameters                 import Parameters   as PM
from   inversion                  import Gaussians    as Integrals
from   PSOworkflow_JointInversion import ParticleSwarmOptimizer as ParticleSwarmOptimizer

noIter    = 200
noPart    = 100
SNR       = 50
noSlice   = 77
randSeed  = 0 

PSO_iter  = 200
MWF_mult  = []
Fit_mult  = []

addNoise  = [False, True]
invT1     = True
invT2     = False
invT2star = False

MWF_intervalls = np.arange(0.025, 0.725, 0.3)
PSO            = ParticleSwarmOptimizer(noPart=noPart,noIter=noIter,SNR=SNR,randSeed=randSeed,
                                        npRand=True,invT1=invT1,invT2star=invT2star,invT2=invT2)

typeSig      = 'T1' if invT1 else 'T2' if invT2 else 'T2_'

###############################################################################
###############################################################################

if invT1==True:
    valMax = PSO.Inversion.T1max
    valMin = PSO.Inversion.T1min
    valMod = PSO.Inversion.modSpace
    
    grid = hlp.make_grid(valMin, valMax, valMod, mode='lin')
    A    = system_matrix_from_kernel(PSO.Inversion.T1_timepoints, grid, T1_decay_abs)

if invT2star==True:
    valMax = PSO.Inversion.T2star_max
    valMin = PSO.Inversion.T2star_min
    valMod = PSO.Inversion.modSpace
    
    grid = hlp.make_grid(valMin, valMax, valMod, mode='lin')    
    A    = system_matrix_from_kernel(PSO.Inversion.T2star_timepoints, grid, T2_decay)

if invT2==True:
    valMax = PSO.Inversion.T2max
    valMin = PSO.Inversion.T2min
    valMod = PSO.Inversion.modSpace
    
    grid        = hlp.make_grid(valMin, valMax, valMod, mode='lin')
    EpyG_params = EpyG_Paramset(PSO.Inversion.T2_T1,PSO.Inversion.T2_flipangle, 
                                PSO.Inversion.T2_TE,PSO.Inversion.T2_ETL)
    
    A           = EpyG_mSE_system_matrix(EpyG_params, grid)

###############################################################################
###############################################################################

for addNoise in addNoise:
    
    obsData_slow = []
    obsData_fast = []
        
    for i, MWF in enumerate(MWF_intervalls):
        # slow solver
        data    = PSO.create_obs_data(MWF=MWF, SNR=SNR, addNoise=addNoise)
        obsData_slow.append(data)
        
        # fast solver
        if invT1 == True:
            values = np.array([100,1000])
            weights = np.array([MWF, 1-MWF]) # integrals of the Gaussians
            widths  = np.array([20, 100])       # standard deviation of the Gaussians
        else:
            values = np.array([8, 80])
            weights = np.array([MWF, 1-MWF]) # integrals of the Gaussians
            widths  = np.array([2, 8])       # standard deviation of the Gaussians
        
        m1   = hlp.gauss(grid, widths[0], values[0], weights[0])
        m2   = hlp.gauss(grid, widths[1], values[1], weights[1])
        
        m    = np.add(m1,m2)
        data = np.matmul(A, m)*(valMax-valMin)/valMod
    
        if addNoise:
            np.random.seed(234567)
            data = data + np.random.normal(loc = 0, scale = data[0] / SNR, size = np.shape(data))
        
        obsData_fast.append(data)
    
    
    obsData_slow = np.column_stack(obsData_slow)
    obsData_fast = np.column_stack(obsData_fast)
    
    x_lim        = range(len(obsData_slow[:, 0]))
    
    fig, ax = plt.subplots(nrows=1, tight_layout=True)
    
    for i in range(obsData_slow.shape[1]):
        ax.plot(PSO.Inversion.T2epg_timepoints, obsData_slow[:,i], label=f'{MWF_intervalls[i]}_slow')
        ax.plot(PSO.Inversion.T2epg_timepoints, obsData_fast[:,i], label=f'{MWF_intervalls[i]}_fast')
        ax.legend(title='slow vs. fast')
    
    # Angeben des Dateipfads für die CSV-Datei
    savepath = './../grafics/performance_test/fast_vs_slow/'
    os.makedirs(savepath, exist_ok=True)
    
    csv_file_path_slow = f'{savepath}obsData_test_slow_{typeSig}_noise{addNoise}.csv'
    csv_file_path_fast = f'{savepath}obsData_test_fast_{typeSig}_noise{addNoise}.csv'
    
    # CSV-Datei schreiben
    np.savetxt(csv_file_path_slow, obsData_slow, delimiter=',', fmt='%f')
    np.savetxt(csv_file_path_fast, obsData_fast, delimiter=',', fmt='%f')
    
    fig.savefig(f'{savepath}{typeSig}_noise{addNoise}.png', dpi=300, format='png')   

    plt.close()     