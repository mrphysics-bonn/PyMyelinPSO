# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:16:27 2023

@author: kobe
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:32:14 2023

@author: kobe
"""

###############################################################################
# Calculation of sensitivity matrices for each MRI signal - T1, T2, T2* #######
###############################################################################

###############################################################################
# 1. Import Modules
import os, time, sys
import matplotlib.pyplot       as     plt
import numpy                   as     np
import helpTools               as     hlp
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   kernels                 import T1_decay_abs, T2_decay, system_matrix_from_kernel
from   helpToolsT2             import EpyG_Paramset, EpyG_mSE_decay, EpyG_mSE_system_matrix

from   parameters              import Parameters as PM

###############################################################################
# 2. Definition of model parameters
# 2.1. Integration/Inversion within the model space
startTime        = time.time()
param            = PM(noPart=None, noIter=None)

sys.exit()

datSpace         = 24                 # step size for the plot of the syn/obs Data
SNR              = 200                # relatively added noice 1/SNR

T1interv         = (0.01, 2000)       # integration borders
T1_timepoints    = np.linspace(1, 5000, datSpace) # integration timesteps in ms

T2sinterv        = (0.01, 200)        # integration borders
T2s_timepoints   = np.linspace(1, 150, datSpace) # integration timesteps in ms

T2interv         = (0.01, 200)
T2_TE            = 5
T2_flipangle     = 120
T2_ETL           = datSpace
T2_T1            = 1000  
T2epg_timepoints = np.linspace(T2_TE, T2_ETL*T2_TE, T2_ETL)

# 2.2. Sensitivity matrix and others

noParam          = 5
noSignals        = 3
MWF_intervall    = np.arange(0.025, 0.71, 0.075)

for testseed in [0, 600, 80, 122345, 98765, 5673, 11, 12, 13][:1]:
    
    sensitivity_T1  = np.zeros((len(MWF_intervall), noParam, datSpace))
    sensitivity_T2  = np.zeros((len(MWF_intervall), noParam, datSpace))
    sensitivity_T2s = np.zeros((len(MWF_intervall), noParam, datSpace))
    
    for ii, MWFstart in enumerate(MWF_intervall):
        
        sensitivity    = np.zeros((3, noParam, datSpace))
        np.random.seed(testseed)
        
        ###############################################################################    
        # 3. Calculation of synthetic data using the integration based slow method
        # 3.1. Producing the model vectors with random values (ParamT2 == ParamT2*)
        
        MWF            = (0.015, 0.7)
        
        m1_T1          = (50, 300)       # center of the gaussian, 1st peak  
        m1_sig_T1      = (5, 50)         # standard deviation of m1
        m2_T1          = (700, 1300)     # center of the gaussian, 2nd peak 
        m2_sig_T1      = (30, 150)       # standard deviation of m2
        
        m1_T2          = (5, 30)
        m1_sig_T2      = (1, 20)
        m2_T2          = (51, 130)
        m2_sig_T2      = (1, 20)
        
        # m1_rand        = np.zeros(3)
        # m1_rand[0]     = np.random.uniform(m1_T1[0],m1_T1[1])
        # m1_rand[1]     = np.random.uniform(m1_T2[0],m1_T2[1])
        # m1_rand[2]     = np.random.uniform(m1_T2[0],m1_T2[1])
        
        # m1_sig_rand    = np.zeros(3)
        # m1_sig_rand[0] = np.random.uniform(m1_sig_T1[0],m1_sig_T1[1])
        # m1_sig_rand[1] = np.random.uniform(m1_sig_T2[0],m1_sig_T2[1])
        # m1_sig_rand[2] = np.random.uniform(m1_sig_T2[0],m1_sig_T2[1])
        
        # m2_rand        = np.zeros(3)
        # m2_rand[0]     = np.random.uniform(m2_T1[0],m2_T1[1])
        # m2_rand[1]     = np.random.uniform(m2_T2[0],m2_T2[1])
        # m2_rand[2]     = np.random.uniform(m2_T2[0],m2_T2[1])
        
        # m2_sig_rand    = np.zeros(3)
        # m2_sig_rand[0] = np.random.uniform(m2_sig_T1[0],m2_sig_T1[1])
        # m2_sig_rand[1] = np.random.uniform(m2_sig_T2[0],m2_sig_T2[1])
        # m2_sig_rand[2] = np.random.uniform(m2_sig_T2[0],m2_sig_T2[1])
        
        # MWF_ran        = np.zeros(3)
        # MWF_rand[0]    = MWFstart            # np.random.uniform(MWF[0],MWF[1])
        # MWF_rand[1]    = MWFstart            # np.random.uniform(MWF[0],MWF[1])
        # MWF_rand[2]    = MWFstart            # np.random.uniform(MWF[0],MWF[1])
            
        m1_rand        = np.array([100,  8,  8])
        m1_sig_rand    = np.array([20,   2,  2])
        m2_rand        = np.array([1000, 80, 80])
        m2_sig_rand    = np.array([100,  8,  8])
        MWF_rand       = np.full(3, MWFstart)
        
        modVector      = np.zeros((3,5))
        modVector[0]   = np.array([m1_rand[0],m1_sig_rand[0],m2_rand[0],m2_sig_rand[0],MWF_rand[0]])
        modVector[1]   = np.array([m1_rand[1],m1_sig_rand[1],m2_rand[1],m2_sig_rand[1],MWF_rand[1]])
        modVector[2]   = np.array([m1_rand[2],m1_sig_rand[2],m2_rand[2],m2_sig_rand[2],MWF_rand[2]])
        
        # 3.2. Computation of the synthetic signal with the original model vectors
        
        # T1
        T1_values    = np.array([m1_rand[0],     m2_rand[0]])          # centers of the Gaussians
        T1_weights   = np.array([MWF_rand[0],    1-MWF_rand[0]])       # integrals of the Gaussians
        T1_widths    = np.array([m1_sig_rand[0], m2_sig_rand[0]])      # standard deviation of the Gaussians   
        T1_spectrum  = hlp.gauss_sum(T1_values, T1_widths, T1_weights)
        
        T1_sig_save  = hlp.signal_from_distribution(T1_spectrum, T1_timepoints, T1interv[0], T1interv[1], T1_decay_abs)
                
        # T2
        EpyG_params  = EpyG_Paramset(T2_T1, T2_flipangle, T2_TE, T2_ETL)
        grid         = hlp.make_grid(T2interv[0], T2interv[1], modSpace, mode='lin')
        
        T2_values    = np.array([m1_rand[1],     m2_rand[1]])
        T2_weights   = np.array([MWF_rand[1],    1-MWF_rand[1]])
        T2_widths    = np.array([m1_sig_rand[1], m2_sig_rand[1]]) 
        T2_spectrum  = hlp.gauss_sum(T2_values, T2_widths, T2_weights)
                    
        T2_sig_save  = EpyG_mSE_decay(EpyG_params, grid, gaussian=T2_spectrum)[1]
        
        # T2*
        T2s_values   = np.array([m1_rand[2],     m2_rand[2]])
        T2s_weights  = np.array([MWF_rand[2],    1-MWF_rand[2]])
        T2s_widths   = np.array([m1_sig_rand[2], m2_sig_rand[2]])
        T2s_spectrum = hlp.gauss_sum(T2s_values, T2s_widths, T2s_weights)
            
        T2s_sig_save = hlp.signal_from_distribution(T2s_spectrum, T2s_timepoints, T2sinterv[0], T2sinterv[1], T2_decay)
        
        ###############################################################################    
        # 4. Calculation of the sensitivity matrix
        # 4.1. Change the model vector iterative for each entry
        #      --> entries of modVector: m1, m1sig, m2, m2sig, MWF
        #      --> mio_in_percent
        
        modVector_raw  = np.copy(modVector)
        mio_in_percent = 1e-8
        
        for i in range(noParam):
            
            # (a) model vector == model vector as calculated as raw
            modVector       = np.copy(modVector_raw)
            
            # (b) infinitesimal change of the model vectors entries (for each signal)
            modVector[:,i] += modVector[:,i] * mio_in_percent
            
            # (c) recalculate the synthetic signals of MRI
            
            # T1
            T1_values    = np.array([modVector[0,0],   modVector[0,2]])   # centers of the Gaussians
            T1_weights   = np.array([modVector[0,4], 1-modVector[0,4]])   # integrals of the Gaussians
            T1_widths    = np.array([modVector[0,1],   modVector[0,3]])   # standard deviation of the Gaussians   
            
            T1_spectrum  = hlp.gauss_sum(T1_values, T1_widths, T1_weights)    
            T1_signal    = hlp.signal_from_distribution(T1_spectrum, T1_timepoints, T1interv[0], T1interv[1], T1_decay_abs)
                    
            # T2
            EpyG_params  = EpyG_Paramset(T2_T1, T2_flipangle, T2_TE, T2_ETL)
            grid         = hlp.make_grid(T2interv[0], T2interv[1], modSpace, mode='lin')
            
            T2_values    = np.array([modVector[1,0],   modVector[1,2]])
            T2_weights   = np.array([modVector[1,4], 1-modVector[1,4]])
            T2_widths    = np.array([modVector[1,1],   modVector[1,3]]) 
            
            T2_spectrum  = hlp.gauss_sum(T2_values, T2_widths, T2_weights)                
            T2_signal    = EpyG_mSE_decay(EpyG_params, grid, gaussian=T2_spectrum)[1]
            
            # T2*
            T2s_values   = np.array([modVector[2,0],   modVector[2,2]])
            T2s_weights  = np.array([modVector[2,4], 1-modVector[2,4]])
            T2s_widths   = np.array([modVector[2,1],   modVector[2,3]])
            
            T2s_spectrum = hlp.gauss_sum(T2s_values, T2s_widths, T2s_weights)        
            T2s_signal   = hlp.signal_from_distribution(T2s_spectrum, T2s_timepoints, T2sinterv[0], T2sinterv[1], T2_decay)
            
            # (d) compare the recalculate3d synthetic signal with the original synthetic signal
            sensitivity[0,i] = T1_sig_save  - T1_signal
            sensitivity[1,i] = T2_sig_save  - T2_signal
            sensitivity[2,i] = T2s_sig_save - T2s_signal

    ###############################################################################
    # 5. Save for plot
    # # keine Normierung
    #     sensitivity_T1[ii]  = np.abs(np.copy(sensitivity[0]))
    #     sensitivity_T2[ii]  = np.abs(np.copy(sensitivity[1]))
    #     sensitivity_T2s[ii] = np.abs(np.copy(sensitivity[2]))
    # Elementweise Normierung indexbasiert
        sensitivity_T1[ii]  = np.abs(np.copy(sensitivity[0]))/T1_sig_save
        sensitivity_T2[ii]  = np.abs(np.copy(sensitivity[1]))/T2_sig_save
        sensitivity_T2s[ii] = np.abs(np.copy(sensitivity[2]))/T2s_sig_save
    # # Normierung auf 1
    #     sensitivity_T1[ii]  = np.abs(np.copy(sensitivity[0]))/np.sum(np.abs(np.copy(sensitivity[0])))
    #     sensitivity_T2[ii]  = np.abs(np.copy(sensitivity[1]))/np.sum(np.abs(np.copy(sensitivity[1])))
    #     sensitivity_T2s[ii] = np.abs(np.copy(sensitivity[2]))/np.sum(np.abs(np.copy(sensitivity[2])))  
    
    ###############################################################################    
    # 5. Plot sensitivity matrix

    savePath = './performance_doc/sensitivity_matrices/'

    os.makedirs(savePath, exist_ok=True)
    
    cmap     = plt.get_cmap('viridis')
    typeSig  = ['T1', 'T2', 'T2s']
    
    for signal in typeSig:
        
        sensitivity_sig = sensitivity_T1 if signal=='T1' else \
                          sensitivity_T2 if signal=='T2' else \
                          sensitivity_T2s
        
        fig, ax  = plt.subplots(nrows=int(sensitivity_T1.shape[0]/2), ncols=2, 
                               figsize=(14, 16), tight_layout=True)
    
        for jj, matrix in enumerate(sensitivity_sig):
        
            if jj < 5:
                pp     = 0
                jj     = jj 
                valMWF = np.round(MWF_intervall[jj],3);
        
            if jj >= 5: 
                pp     = 1
                jj     = jj-5
                valMWF = np.round(MWF_intervall[jj+5],3)
        
            im = ax[jj,pp].imshow(matrix, cmap=cmap, interpolation='nearest')#,
                                  #vmin=0, vmax=np.max(sensitivity_sig))
            
            ax[jj,pp].set_title(rf'MWF: {valMWF:.3f}')
            ax[jj,pp].set_yticks(range(matrix.shape[0]), [])
            
            ax[jj,pp].set_xticks([], [])
        
            divider = make_axes_locatable(ax[jj, pp])
            cax = divider.append_axes("bottom", size="15%", pad=0.3)  # Position der Farbleiste
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        
        fig.suptitle(rf'{signal}  | MWF: {MWF_intervall[0]:.3f}-{MWF_intervall[-1]:.3f} |  '
                     rf'Δ$\vec{{m}}$: {mio_in_percent} * $\vec{{m}}$ [%]', fontsize=16);
            
        fig.savefig(savePath+f'{signal}_{mio_in_percent}.jpg', dpi=200, format='jpg')
        plt.close(fig)