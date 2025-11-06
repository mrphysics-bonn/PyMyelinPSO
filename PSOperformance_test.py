# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:32:05 2025

@author: kobe
"""

# Tests

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:38:17 2025

@author: kobe
"""

###############################################################################
# Script create synthetic relaxation curves for T2 and T2*  ###################
###############################################################################

# import of necessary tools: built-in, installed
import timeit, os, re, sys, glob, warnings
import numpy               as     np
from   pathlib             import Path
from   numba               import cuda, njit
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# import of necessary tools: own scripts and classes
# have to be stored in the same directory as the main script
import helpTools   as     hlp
from   PSOworkflow import ParticleSwarmOptimizer
from   PSOplots    import PSOvideos
from   mwf_t1t2t2s import mwf_analysis

warnings.filterwarnings("ignore", category=RuntimeWarning)

# config file and paths
config_path = r'F:\JIMM2\MWF_invivo\Python_V.1.0.5\Denoising\config_denoising.yaml'
config      = hlp.open_parameters(path=config_path)
data_dir    = os.path.normpath(os.path.join(config['source']['file']['prj_dir'], 'nifti'))

# model parameters for creating gaussian curves (look at the tutorial)
_m1     = (5,   45)                           # center of the gaussian, 1st peak 
_m1_sig = (0.1, 100)                          # standard deviation of m1
_m2     = (60,  85)                           # center of the gaussian, 2nd peak 
_m2_sig = (0.1, 100)                          # standard deviation of m2
_m3     = (80,  110)                          # center of the gaussian, 3rd peak
_m3_sig = (0.1, 100)                          # standard deviation of m2
_int2   = (0.1, 5)                            # area under the curve of m2 gaussian
_int3   = (0.1, 5)                            # area under the curve of m3 gaussian
_MWF    = (0,   0.85)                         # typical intervall of MWF in a MRI
_MW_f   = (-75, 75)                           # frequency shift MW component (Hz)
_EW_f   = (-25, 25)                           # frequency shift EW component (Hz)
_AW_f   = (-25, 25)                           # frequency shift AW component (Hz)                
_phi    = (-15, 15)                           # global phase shift

noCurves = 400                                 # number of synthetic relaxation curves
noSlice  = 12                                 # no of slice for matrix definition
signal   = 'T2'                               # 'T2', 'T2*'
phase    = False                              # for T2* signal only

T2min    = 0
T2max    = 200
modSpace = 1000

xx, yy   = 55, 45                             # pixel
np.random.seed(seed=0)
width    = 6.5

partial_test_1    = False 
partial_test_2    = False
sigma_test_1      = False    # sigma threshold for distinguishing gauss_full/gauss_part
sigma_test_2      = False
batch_size_test_1 = False    # batch process of different sizes on partial gauss calculation
batch_size_test_2 = False

        
# initalizing of the PSO class for creating synthetic data
# --> PSO.createSynData_fast()
PSOclass = ParticleSwarmOptimizer(config_data=config)

###############################################################################
# creating MWF_Analysis class instance (Tony Stoecker from DZNE Bonn)  
rootMWF     = mwf_analysis(data_dir = data_dir,
                           KW_B1    = config['source']['file']['B1'],
                           KW_T1    = config['source']['file']['T1'],
                           KW_T2    = config['source']['file']['T2'],
                           KW_T2S   = config['source']['file']['T2S'],
                           KW_T2SP  = config['source']['file']['T2SP'],
                           T2S_TE   = config['source']['file']['TE'])

PSOclass.init_MWF_Analysis(rootMWF, noSlice)

n_fs       = np.argmin(np.abs(rootMWF.b1_grid-rootMWF.data.slice['B1'][yy,xx]))        
sys_param_ = {'T2_MATRIX':  rootMWF.sm['T2'][:,:,n_fs], 'T2S_MATRIX': rootMWF.sm['T2S'][:,:,0]}

# define the system matrices for relaxation signals
PSOclass.init_sysMatrix(sys_param_)

###############################################################################
# calculation of gaussians
# @njit
def gauss_full(x, sigma, mean, scale):
    '''
    args:
        x - [1D array] integration step size
    '''
    return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))  # 15.326399996411055 seconds

def gauss_part(x, sigma, mean, scale, acc=10**-10):
    
    '''
    Creates the very same object as gauss_full(), but uses only maximum peak-intervals.
    
    args:
        x - system grid - and array which ...
    '''
    
    threshold      = np.log(acc)                                    # 0.026999972760677338 seconds
    array          = np.zeros((sigma.shape[0], x.shape[1]))         # 0.1142999972216785 seconds   
    exp_arg        = -np.square(x - mean) / (2 * sigma ** 2)        # 7.850399997550994 seconds    
    mask           = exp_arg >= threshold                           # 0.5229000234976411 seconds
    exp_arg_ind    = np.where(mask)[1]                              # 1.0704000014811754 seconds
    idx_gt, idx_lt = np.min(exp_arg_ind), np.max(exp_arg_ind)+1     # 2*0.2930000191554427 seconds
    
    array[:, idx_gt:idx_lt] = scale / (np.sqrt(2*np.pi)*sigma) * np.exp(exp_arg[:,idx_gt:idx_lt]) # 6.513000000268221 seconds
        
    return array

def gauss_part_test(x, sigma, mean, scale, acc=10**-10, width=6.5, chunk_size=500):
    
    ''' creating gauss part using width of the gaussian at lim(y)-->0 '''
    
    array     = np.zeros((sigma.shape[0], x.shape[1]))                    # 0.1142999972216785 seconds    
    mean_min  = np.min(mean)
    mean_max  = np.max(mean)
    sigma_max = np.max(sigma)
    
    indices = np.where((x[0,:] > mean_min-width*sigma_max) & 
                       (x[0,:] < mean_max+width*sigma_max))[0]  # 0.10769994696602225 seconds
    
    exp_arg = -((x[:, indices[0]:indices[-1]+1] - mean)**2) / (2 * sigma ** 2)          # 6.236499990336597 seconds    
   
    array[:, indices[0]:indices[-1]+1] = scale / (np.sqrt(2*np.pi)*sigma) * np.exp(exp_arg) # 6.08929997542873 seconds
    
    return array 

def gauss_part_test2(x, sigma, mean, scale, acc=10**-10, width=6.5):

    # mean_min  = np.min(mean, axis=1)
    # mean_max  = np.max(mean, axis=1)
    # sigma_max = np.max(sigma, axis=1)   

    # left  = np.searchsorted(x[0], mean_min - width * sigma_max, side='left')
    # right = np.searchsorted(x[0], mean_max + width * sigma_max, side='right')
    
    # x_sub_0 = x[0, left[0, 0]:right[0, 0]]
    # x_sub_1 = x[0, left[1, 0]:right[1, 0]]
    # x_sub_2 = x[0, left[2, 0]:right[2, 0]]
    
    # x_sub = np.array([x_sub_0, x_sub_1, x_sub_2], dtype=object)
    
    # mean_min  = np.min(mean)
    # mean_max  = np.max(mean)
    # sigma_max = np.max(sigma)
    
    left  = np.searchsorted(x[0], np.min(mean) - width * np.max(sigma), side='left')
    right = np.searchsorted(x[0], np.max(mean) + width * np.max(sigma), side='right')
    x_sub = x[0, left:right]

    exp_arg = -np.square(x_sub - mean) / (2 * sigma**2)
    factor  = scale / (np.sqrt(2 * np.pi) * sigma)
    
    array = np.zeros((sigma.shape[0], x.shape[1]))
    array[:, left:right] = factor * np.exp(exp_arg)

    return array

# @cuda.jit
# def gauss_part_cuda(x_sub, sigma, mean, scale, output):
#     i = cuda.grid(1)
#     if i < mean.shape[0]:
#         s = sigma[i]
#         m = mean[i]
#         sc = scale[i]
#         for j in range(x_sub.shape[0]):
#             arg = -(x_sub[j] - m) ** 2 / (2 * s ** 2)
#             factor = sc / (np.sqrt(2 * np.pi) * s)
#             output[i, j] = factor * np.exp(arg)
    
# output = np.zeros((400, 1000), dtype=np.float32)
# threads_per_block = 32
# blocks_per_grid = (mean.shape[0] + threads_per_block - 1) // threads_per_block

# gauss_part_cuda[blocks_per_grid, threads_per_block](x_sub, sigma, mean, scale, output)

def batch_process_part_test(size):
    batch_size = size
    num_batches = noCurves // batch_size
    
    # values  = np.array([m1, m2])                        # x-value of gaussian peak
    # weights = np.array([int2*MWF/(1-MWF), int2])        # area beyond gaussian peak
    # widths  = np.array([m1_sig, m2_sig]) 

    for i in range(num_batches):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        
        gauss_part_test2(PSOclass.sysGrid['T2S'][None, :], widths[0, idx, None], values[0, idx, None], weights[0, idx, None])

def batch_process_full(size):
    batch_size = size
    num_batches = noCurves // batch_size
    
    # values  = np.array([m1, m2])                        # x-value of gaussian peak
    # weights = np.array([int2*MWF/(1-MWF), int2])        # area beyond gaussian peak
    # widths  = np.array([m1_sig, m2_sig]) 

    for i in range(num_batches):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        
        gauss_full(PSOclass.sysGrid['T2S'][None, :], widths[0, idx, None], values[0, idx, None], weights[0, idx, None])    
    
def batch_process_part(size):
    batch_size = size
    num_batches = noCurves // batch_size
    
    # values  = np.array([m1, m2])                        # x-value of gaussian peak
    # weights = np.array([int2*MWF/(1-MWF), int2])        # area beyond gaussian peak
    # widths  = np.array([m1_sig, m2_sig]) 

    for i in range(num_batches):
        idx = slice(i * batch_size, (i + 1) * batch_size)
        
        gauss_part_test(PSOclass.sysGrid['T2S'][None, :], widths[0, idx, None], values[0, idx, None], weights[0, idx, None])

############################################################################################
# partial tests:

if partial_test_1 == True:

    sig = 1

    m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
    m1_sig = np.random.uniform(_m1_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
    m2_sig = np.random.uniform(_m2_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
    m3_sig = np.random.uniform(_m3_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
    int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
    MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
    EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
    phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))
    
    mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)
    
    values  = np.array([m1, m2, m3])                          # x-value of gaussian peak
    weights = np.array([(int2+int3)*MWF/(1-MWF), int2, int3]) # area beyond gaussian peak
    widths  = np.array([m1_sig, m2_sig, m3_sig])              # width of gaussian bell curve
    
    x = PSOclass.sysGrid['T2'][None, :]
    sigma = widths[0, :, None]
    mean  = values[0, :, None]
    scale = weights[0, :, None]
    
    m1_peak = gauss_full(x, sigma, mean, scale)
    
    sigma = widths[1, :, None]
    mean  = values[1, :, None]
    scale = weights[1, :, None]
    
    m2_peak = gauss_full(x, sigma, mean, scale)
    
    sigma = widths[1, :, None]
    mean  = values[1, :, None]
    scale = weights[1, :, None]
    
    m3_peak = gauss_full(x, sigma, mean, scale)
    
    # @njit(cache=False)
    # def gauss_full_njit(x, sigma, mean, scale):
    #     '''
    #     args:
    #         x - [1D array] integration step size
    #     '''
    #     return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))  # 15.326399996411055 seconds

    # execution_time_full = timeit.timeit("gauss_full(x, sigma, mean, scale)",
    #     globals=globals(),number=800)

    # execution_time_njit_init = timeit.timeit("gauss_full_njit(x, sigma, mean, scale)",
    #     globals=globals(),number=1)

    # execution_time_njit = timeit.timeit("gauss_full_njit(x, sigma, mean, scale)",
    #     globals=globals(),number=800)

    # print(f'full, njit_init, njit - {execution_time_full:.5}, {execution_time_njit_init:.5}, {execution_time_njit:.5}')

    @njit(cache=False)
    def fast_matmul(a, b):
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        return a @ b # or np.dot
    
    execution_time_full = timeit.timeit("np.matmul(m1_peak+m2_peak, PSOclass.sysMatrix['T2'].T)",
        globals=globals(),number=100)

    execution_time_njit_init = timeit.timeit("fast_matmul(m1_peak+m2_peak, PSOclass.sysMatrix['T2'].T)",
        globals=globals(),number=1)

    execution_time_njit = timeit.timeit("fast_matmul(m1_peak+m2_peak, PSOclass.sysMatrix['T2'].T)",
        globals=globals(),number=100)

    print(f'full, njit_init, njit - {execution_time_full:.5}, {execution_time_njit_init:.5}, {execution_time_njit:.5}')

    del fast_matmul#; del gauss_full_njit
    
    import gc
    gc.collect()

if partial_test_2 == True:
    
    sig = 20

    m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
    m1_sig = np.random.uniform(_m1_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
    m2_sig = np.random.uniform(_m2_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
    m3_sig = np.random.uniform(_m3_sig[0], sig,          noCurves)#size=(110,110,noCurves))
    int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
    int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
    MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
    EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
    phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))
    
    mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)
    
    values  = np.array([m1, m2, m3])                          # x-value of gaussian peak
    weights = np.array([(int2+int3)*MWF/(1-MWF), int2, int3]) # area beyond gaussian peak
    widths  = np.array([m1_sig, m2_sig, m3_sig])              # width of gaussian bell curve
    
    x = PSOclass.sysGrid['T2'][None, :]
    sigma = widths[1, :, None]
    mean  = values[1, :, None]
    scale = weights[1, :, None]
    
    indices = np.where((x[0,:] > np.min(mean)-width*np.max(sigma)) & 
                       (x[0,:] < np.max(mean)+width*np.max(sigma)))[0]

    exp_arg = -((x[:, indices[0]:indices[-1]+1] - mean)**2) / (2 * sigma ** 2)
    
    for ind in range(0,500,20):
        time_sqrt = timeit.timeit(
        "-np.square(x[:, 0:ind] - mean) / (2 * sigma ** 2)", globals=globals(), number=1000)
        
        time_exp = timeit.timeit(
        "scale / (np.sqrt(2*np.pi)*sigma) * np.exp(exp_arg[:, 0:ind])", globals=globals(), number=1000)

        if isinstance(sig, int):
            _sig = f'{sig:3d}'
        else:
            _sig = f'{sig:1.1f}'
        
        print(f'{ind}: [sqrt, exp] | {time_sqrt:.4f}, {time_exp:.4f} sec')

############################################################################################
# sigma test:  - sigma threshold for distinguishing gauss_full/gauss_part

if sigma_test_1 == True:
    
    for i in [0,1,2]:
    
        # for sig in [0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50]:
        for sig in [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:
            
            m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
            m1_sig = np.random.uniform(_m1_sig[0], sig,          noCurves)#size=(110,110,noCurves))
            m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
            m2_sig = np.random.uniform(_m2_sig[0], sig,          noCurves)#size=(110,110,noCurves))
            m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
            m3_sig = np.random.uniform(_m3_sig[0], sig,          noCurves)#size=(110,110,noCurves))
            int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
            int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
            MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
            MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
            EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
            MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
            phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))
            
            mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)
            
            values  = np.array([m1, m2, m3])                          # x-value of gaussian peak
            weights = np.array([(int2+int3)*MWF/(1-MWF), int2, int3]) # area beyond gaussian peak
            widths  = np.array([m1_sig, m2_sig, m3_sig])              # width of gaussian bell curve
            
            execution_time_full = timeit.timeit(
                "gauss_full(PSOclass.sysGrid['T2'][None, :], widths[i, :, None], values[i, :, None], weights[i,:, None])",
                globals=globals(), number=200)
            
            execution_time_part = timeit.timeit(
                "gauss_part(PSOclass.sysGrid['T2'][None, :], widths[i, :, None], values[i, :, None], weights[i,:, None])",
                globals=globals(), number=200)
            
            execution_time_part_test = timeit.timeit(
                "gauss_part_test(PSOclass.sysGrid['T2'][None, :], widths[i, :, None], values[i, :, None], weights[i,:, None])",
                globals=globals(), number=200)
    
            execution_time_part_test2 = timeit.timeit(
                "gauss_part_test2(PSOclass.sysGrid['T2'][None, :], widths[i, :, None], values[i, :, None], weights[i,:, None])",
                globals=globals(), number=200)
            
            if isinstance(sig, int):
                _sig = f'{sig:3d}'
            else:
                _sig = f'{sig:1.1f}'
            
            print(f"m{i+1} | sig = {_sig} | [full, part, part_test_1, part_test_2] -> {execution_time_full:.4f}, {execution_time_part:.4f}, {execution_time_part_test:.4f},  {execution_time_part_test2:.4f}] seconds")

        input('continue? [ENTER]')

############################################################################################
# sigma b: calculation of gauss partial test function dependent on sigma
    
if sigma_test_2 == True:
    
    m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
    m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
    m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
    int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
    int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
    MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
    EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
    MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
    phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))

    for sig in [0.1,0.5,1,2,3,4,5,10]:
        m1_sig = np.random.uniform(_m1_sig[0], sig,          noCurves)#size=(110,110,noCurves))
        m2_sig = np.random.uniform(_m2_sig[0], sig,          noCurves)#size=(110,110,noCurves))
        m3_sig = np.random.uniform(_m3_sig[0], sig,          noCurves)#size=(110,110,noCurves))
        
        mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)

        values  = np.array([m1, m2, m3])                          # x-value of gaussian peak
        weights = np.array([(int2+int3)*MWF/(1-MWF), int2, int3]) # area beyond gaussian peak
        widths  = np.array([m1_sig, m2_sig, m3_sig])
        
        x = PSOclass.sysGrid['T2'][None, :]
        sigma = widths[2, :, None]
        mean  = values[2, :, None]
        scale = weights[2, :, None]
        
        array   = np.zeros((sigma.shape[0], x.shape[1]))
        width   = 6.5

        if isinstance(sig, int):
            _sig = f'{sig:3d}'
        else:
            _sig = f'{sig:1.1f}'
        
        time_ind = timeit.timeit(
        "np.where((x[0,:] > np.min(mean)-width*np.max(sigma)) & (x[0,:] < np.max(mean)+width*np.max(sigma)))[0]", globals=globals(), number=1000)

        indices = np.where((x[0,:] > np.min(mean)-width*np.max(sigma)) & 
                           (x[0,:] < np.max(mean)+width*np.max(sigma)))[0]

        # index_string = 'ind_low_high_diff:'
        # print(f'{index_string:<25}{indices[0]}, {indices[-1]}, {indices[-1]-indices[0]}')

        time_sqrt = timeit.timeit(
        "-np.square(x[:, indices[0]:indices[-1]+1] - mean) / (2 * sigma ** 2)", globals=globals(), number=1000)
        
        exp_arg = -np.square(x[:, indices[0]:indices[-1]+1] - mean) / (2 * sigma ** 2)
        
        time_gauss = timeit.timeit(
        "scale / (np.sqrt(2*np.pi)*sigma) * np.exp(exp_arg)", globals=globals(), number=1000)

        print(f"sig = {_sig} | [ind, sqrt, gauss] -> {time_ind:.4f}, {time_sqrt:.4f}, {time_gauss:.4f} sec | ind: {indices[0]}, {indices[-1]}, {indices[-1]-indices[0]}")

        # time_ind = 'time_indices (1000*):'
        # print(f'{ time_string:<25}{1000*time:.4f} seconds')

        # time_sqrt = 'time_square (1000*):'
        # print(f'{ time_string:<25}{1000*time:.4f} seconds')
        
        # time_string = 'time_array_calc (1000*):'
        # print(f'{ time_string:<25}{1000*time:.4f} seconds')

###############################################################################    
# batch size test a:  optimal batch size for handling np.exp() in the gauss function?

if batch_size_test_1 == True:

    print("Test: Batch process of different sizes/sigmas on partial gauss calculation")
      
    for sigma in [0.1, 0.5, 1, 5, 10, 20, 30]:
        
        print('\n')
        
        for size in [2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 400]:
    
            m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
            m1_sig = np.random.uniform(_m1_sig[0],  sigma,     noCurves)#size=(110,110,noCurves))
            m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
            m2_sig = np.random.uniform(_m2_sig[0],  sigma,     noCurves)#size=(110,110,noCurves))
            m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
            m3_sig = np.random.uniform(_m3_sig[0],  sigma,     noCurves)#size=(110,110,noCurves))
            int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
            int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
            MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
            MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
            EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
            MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
            phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))

            mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)
        
            values  = np.array([m1, m2])                        # x-value of gaussian peak
            weights = np.array([int2*MWF/(1-MWF), int2])        # area beyond gaussian peak
            widths  = np.array([m1_sig, m2_sig])                # width of gaussian bell curve
    
            execution_time_part = timeit.timeit(
                "batch_process_full(size=size)",globals=globals(),number=500)
            
            execution_time_part_test = timeit.timeit(
                "batch_process_part(size=size)",globals=globals(),number=500)
        
            print(f"ExTime: runs={200} | sig={sigma} | size={str(size).zfill(3)} | --> {execution_time_part:.4f}, {execution_time_part_test:.4f} sec")
            
###############################################################################    
# batch size test b:  optimal batch size for handling np.matmul()

if batch_size_test_2 == True:
    
    for size in [20,20,25,25,50,50,100,100,200,200]:
        
        m1     = np.random.uniform(_m1[0],     _m1[1],     noCurves)#size=(110,110,noCurves))
        m1_sig = np.random.uniform(_m1_sig[0], 0.1,          noCurves)#size=(110,110,noCurves))
        m2     = np.random.uniform(_m2[0],     _m2[1],     noCurves)#size=(110,110,noCurves))
        m2_sig = np.random.uniform(_m2_sig[0], 0.1,          noCurves)#size=(110,110,noCurves))
        m3     = np.random.uniform(_m3[0],     _m3[1],     noCurves)#size=(110,110,noCurves))
        m3_sig = np.random.uniform(_m3_sig[0], 0.1,          noCurves)#size=(110,110,noCurves))
        int2   = np.random.uniform(_int2[0],   _int2[1],   noCurves)#size=(110,110,noCurves))
        int3   = np.random.uniform(_int3[0],   _int3[1],   noCurves)#size=(110,110,noCurves))
        MWF    = np.random.uniform(_MWF[0],    _MWF[1],    noCurves)#size=(110,110,noCurves))
        MW_f   = np.random.uniform(_MW_f[0],   _MW_f[1],   noCurves)#size=(110,110,noCurves))
        EW_f   = np.random.uniform(_EW_f[0],   _EW_f[1],   noCurves)#size=(110,110,noCurves))
        MW_f   = np.random.uniform(_AW_f[0],   _AW_f[1],   noCurves)#size=(110,110,noCurves))
        phi    = np.random.uniform(_phi[0],    _phi[1],    noCurves)#size=(110,110,noCurves))
        
        mod    = np.stack((m1, m1_sig, m2, m2_sig, int2, MWF), axis=-1)
        
        values  = np.array([m1, m2, m3])                          # x-value of gaussian peak
        weights = np.array([(int2+int3)*MWF/(1-MWF), int2, int3]) # area beyond gaussian peak
        widths  = np.array([m1_sig, m2_sig, m3_sig])              # width of gaussian bell curve
        
        m1_peak = gauss_full(PSOclass.sysGrid['T2'][None, :], widths[0, :, None], values[0, :, None], weights[0, :, None])  
        m2_peak = gauss_full(PSOclass.sysGrid['T2'][None, :], widths[1, :, None], values[1, :, None], weights[1, :, None])  
        m3_peak = gauss_full(PSOclass.sysGrid['T2'][None, :], widths[2, :, None], values[2, :, None], weights[2, :, None])
        
        m_total = np.zeros((noCurves, 1000))
        
        def batch_size_matmul_1(size):
            batch_size = size
            num_batches = noCurves // batch_size
    
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                m_total = m1_peak[idx]  + m2_peak[idx]  + m3_peak[idx] 
                np.matmul(m_total,PSOclass.sysMatrix['T2'].T)*(PSOclass.Inversion.T2Smax-PSOclass.Inversion.T2Smin)/PSOclass.Inversion.modSpace
        
        def batch_size_matmul_2(size):
            batch_size = size
            num_batches = noCurves // batch_size
    
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                np.matmul(m1_peak[idx]  + m2_peak[idx]  + m3_peak[idx] ,PSOclass.sysMatrix['T2'].T)*(PSOclass.Inversion.T2Smax-PSOclass.Inversion.T2Smin)/PSOclass.Inversion.modSpace
                
        def batch_size_matmul_3(size):
            batch_size = size
            num_batches = noCurves // batch_size
    
            for i in range(num_batches):
                idx = slice(i * batch_size, (i + 1) * batch_size)
                m_total[idx] = m1_peak[idx]  + m2_peak[idx]  + m3_peak[idx]
                np.matmul(m_total[idx],PSOclass.sysMatrix['T2'].T)*(PSOclass.Inversion.T2Smax-PSOclass.Inversion.T2Smin)/PSOclass.Inversion.modSpace
 
    
        # def no_batch_matmul():       
        #     np.matmul(m_total,PSOclass.sysMatrix['T2'].T)*(PSOclass.Inversion.T2Smax-PSOclass.Inversion.T2Smin)/PSOclass.Inversion.modSpace
                
        execution_time_1 = timeit.timeit(
            "batch_size_matmul_1(size)",
            globals=globals(), number=200)
        
        execution_time_2 = timeit.timeit(
            "batch_size_matmul_2(size)",
            globals=globals(), number=200)
        
        execution_time_3 = timeit.timeit(
            "batch_size_matmul_3(size)",
            globals=globals(), number=200)
        
        print(f"ExTime batch test np.matmul: runs=100 | BTsize={size} -> {execution_time_1:.4f}; {execution_time_2:.4f}; {execution_time_3:.4f} seconds")
        
        # if size == 400:
        #     execution_time_full = timeit.timeit(
        #         "batch_size_matmul(size)",
        #         globals=globals(), number=200)
            
        #     print(f"ExTime batch test np.matmul: runs=100 | BTsize=full -> {execution_time_full} seconds")

############################################################################################
### useful tools

def mask_size_comparison():
    mask_path_a = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_a\mask\mask_refined_24_10_10_a.npy'
    mask_path_b = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_b\mask\mask_refined_24_10_10_b.npy'
    mask_path_c = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\mask\mask_refined_24_10_10_c.npy'
    
    masks_a = np.load(file=mask_path_a, allow_pickle=True)
    masks_b = np.load(file=mask_path_b, allow_pickle=True)
    masks_c = np.load(file=mask_path_c, allow_pickle=True)
    
    for i in range(24):
        print(f'slice {str(i).zfill(2)}: {len(np.argwhere(masks_a[:,:,i]))}, {len(np.argwhere(masks_b[:,:,i]))}, {len(np.argwhere(masks_c[:,:,i]))}')


def refine_mask(subject='b', signal='T2', selection_mode='both'):
    
    config_path = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\config_24_10_10_{subject}.yaml'
    config      = hlp.open_parameters(path=config_path)

    mask_path_b = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\mask\mask_refined_24_10_10_{subject}.npy'
    masks_b     = np.load(file=mask_path_b, allow_pickle=True)

    data_dir    = os.path.normpath(f'{config["source"]["file"]["prj_dir"]}/nifti/')
    data_file   = config["source"]["file"][signal]

    img_nifti   = hlp.load_data(data_dir, data_file)[0]
    img_data    = img_nifti.get_fdata()    
    img_data    = np.sum(img_data, axis=-1)                              # sum over 4th axis
    
    mask_refined = np.empty_like(img_data)

    for layer in range(img_data.shape[-1]):
        array   = hlp.mask_shape(binary_array     = masks_b[:,:,layer], 
                                 comparison_array = img_data[:,:,layer], 
                                 layer=layer, transparency=0.5, selection_mode=selection_mode)
        
        mask_refined[:,:,layer] = array
    
    return mask_refined

def show_meas_data(subject='c', slice_=11, signal='T2', norm=None):

    config_path = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\config_24_10_10_{subject}.yaml'
    
    config      = hlp.open_parameters(path=config_path)
    data_dir    = os.path.normpath(os.path.join(config['source']['file']['prj_dir'], 'nifti'))

    _PSO        = ParticleSwarmOptimizer(config_data=config)
    _slice      = slice_

    norm        = norm

    # rootMWF has to be called here, otherwise, signal data is overwritten
    rootMWF     = mwf_analysis(data_dir = data_dir,
                               KW_B1    = config['source']['file']['B1'],
                               KW_T1    = config['source']['file']['T1'],
                               KW_T2    = config['source']['file']['T2'],
                               KW_T2S   = config['source']['file']['T2S'],
                               KW_T2SP  = config['source']['file']['T2SP'],
                               T2S_TE   = config['source']['file']['TE'])

    _PSO.init_MWF_Analysis(rootMWF, _slice)
    
    import matplotlib.pyplot as plt
    from   matplotlib.cm import get_cmap

    coords = [[5, 5], [45, 50], [70, 70]]
    col_bl = get_cmap("Blues")(np.linspace(0.4, 1, 3)) 
    col_rd = get_cmap("Reds")(np.linspace(0.4, 1, 3)) 
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), tight_layout=True)

    for i,(y,x) in enumerate(coords):
        curve = rootMWF.data.slice['T2'][y,x]/np.max(rootMWF.data.slice['T2'][y,x])
        ax[1].plot(curve, color=col_bl[i], label=f"T2 @ [{y},{x}]")

    for i,(y,x) in enumerate(coords):
        curve = rootMWF.data.slice['T2S'][y,x]/np.max(rootMWF.data.slice['T2S'][y,x])
        ax[2].plot(curve, color=col_rd[i], label=f"T2S @ [{y},{x}]")

    ax[0].imshow(rootMWF.data.slice['T2'][:,:,0])

    for i,(y,x) in enumerate(coords):
        ax[0].scatter(y, x, color='red')

    ax[1].legend(fontsize=12)
    ax[2].legend(fontsize=12)
    plt.show()
        
def comp_meas_data_MWF_PSO(subject='c', slice_=11, signal='T2', pixel=[45,50]):
    
    ''' Signal in ['T1', 'T2', 'T2S', 'T2SP'] '''
    
    config_path = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\config_24_10_10_{subject}.yaml'
    config      = hlp.open_parameters(path=config_path)
    data_dir    = os.path.normpath(os.path.join(config['source']['file']['prj_dir'], 'nifti'))
    
    config['source']['signal'].update({sig: False for sig in ['T1', 'T2', 'T2S', 'T2SP']})
    config['source']['signal'][signal] = True
        
    if signal == 'CT2S': 
        config['source']['signal']['T2S'] = True;  config['source']['signal']['T2SP'] = True

    _PSO        = ParticleSwarmOptimizer(config_data=config)
    _slice      = slice_

    # rootMWF has to be called here, otherwise, signal data is overwritten
    rootMWF     = mwf_analysis(data_dir = data_dir,
                               KW_B1    = config['source']['file']['B1'],
                               KW_T1    = config['source']['file']['T1'],
                               KW_T2    = config['source']['file']['T2'],
                               KW_T2S   = config['source']['file']['T2S'],
                               KW_T2SP  = config['source']['file']['T2SP'],
                               T2S_TE   = config['source']['file']['TE'])

    _PSO.init_MWF_Analysis(rootMWF, _slice)
    
    obsData  = _PSO.getObsDataFromMeas(position = (pixel[0],pixel[1],slice_),
                                       data_dir = data_dir,
                                       pathT1   = config['source']['file']['T1'],
                                       pathT2   = config['source']['file']['T2'],
                                       pathT2S  = config['source']['file']['T2S'],
                                       pathT2SP = config['source']['file']['T2SP'],
                                       pathTE   = config['source']['file']['TE'])

    dataPSOclass  = obsData[signal]
    dataMWF_pixel = rootMWF.data.slice[signal][pixel[0],pixel[1]]
    
    if not signal == 'CT2S':
        dataMWFclass  = dataMWF_pixel/np.max(dataMWF_pixel)
    else:
        dataMWFclass  = dataMWF_pixel
    
    print(dataMWFclass, dataPSOclass, np.array_equal(dataMWFclass, dataPSOclass))

###############################################################################
### comparing results from different PSO program versions

def comp_MWF_array(filelist: list, signal: str, sigma: int, no_peaks: str, server: str, versions: tuple):

    if signal == 'CT2S': _signal = 'T2S'
    else:                _signal = signal
        
    # vs_files = [re.search(r'v_\d+[A-Za-z]?\b', p).group() for p in filelist]
    vs_files = [re.search(r'v_\d+[A-Za-z]?(?:_\d+)?\b', p).group() for p in filelist]

    if versions[1] not in vs_files: return
    
    PSOdic   = {vs_files[i]: np.load(f, allow_pickle=True).item()[_signal] for i,f in enumerate(filelist)}  # filelist = Liste von .npy-Dateien
    
    # test if versions areequal
    # items are:
    # GAUSS results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: int2 | 5: MWF    | -1: misfit
    # DIRAC results - 0: m1   | 1: m1_sig | 2:  m2  | 3:  m2_sig | 4: m3   | 5: m3_sig | 
    #                 6: int2 | 7: int3   | 8: MWF  | -1: misfit
    
    # create a mask based on MWF value and best map in array [-1]
    no_peak, ind_MWF  = (2,5) if no_peaks == 'GAUSS' else (3,8)
    try:
        mask              = ~np.isnan(PSOdic[versions[0]][ind_MWF,:,:,-1])
    except:
        return
    
    basic_version_MWF = PSOdic[versions[0]][ind_MWF,:,:,-1][mask]
    
    print(f'Signal: {signal} | peaks: {no_peak} | sigma: {sigma} | reference: {versions[0]} | server: {server}')
    
    for key in versions[1:]:
        title_string = [signal, sigma, no_peaks, key]
        comp_version_MWF = PSOdic[key][ind_MWF,:,:,-1][mask]
        match_up_arrays(basic_version_MWF, comp_version_MWF, threshold=10**-2, string=key, plot=[True,'',mask, title_string])
         
def match_up_arrays(a: np.array, b: np.array, threshold: float, string='', plot=[False, '', np.empty(0), list]):

    diff = np.abs(a - b)

    # if equal
    if diff.size == 0 or np.all(diff == 0):
        print(f"{string}: arrays are even (incl. NaNs).\n")
        return
    else:
        max_diff = np.max(diff)
        # Berechne Anzahl gemeinsamer Nachkommastellen (dezimalen Genauigkeit)
        decimal_match = int(-np.floor(np.log10(max_diff)))
        # print(f"Max. Abweichung: {max_diff:.2e}")
        if plot[0]==False:
            print(f"{string}: arrays match up to the {decimal_match-1}th decimal place.\n")
    
    if plot[0]==True:        
                  
        # array = np.zeros([110,110,4])
        array = np.full((110,110,4), np.nan)
        array[plot[2],0], array[plot[2],1], array[plot[2],2] = a, b, diff
        
        mask_diff               = diff > threshold
        full_mask_diff          = np.zeros_like(plot[2], dtype=bool)
        full_mask_diff[plot[2]] = mask_diff
                                          
        array[full_mask_diff,3] = 1
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), tight_layout=True)
        cmap    = plt.cm.viridis.copy()
        cmap.set_bad(color='black')
        
        im1 = ax[0,0].imshow(np.rot90(array[:,:,0]), vmin=0, vmax=0.35, cmap=cmap)
        fig.colorbar(im1, ax=ax[0,0])
        
        im2 = ax[0,1].imshow(np.rot90(array[:,:,1]), vmin=0, vmax=0.35, cmap=cmap)
        fig.colorbar(im2, ax=ax[0,1])
                
        maxval = np.nanmax(array[:,:,2])
        # if maxval > 0:
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         logval = np.log10(maxval)
        #     decimal = int(-np.floor(logval))
        #     rounded = round(maxval, decimal)
        # else:
        #     rounded = 0
            
        try:
            decimal = int(-np.floor(np.log10(abs(np.nanmax(array[:,:,2])))))
            rounded = round(np.nanmax(array[:,:,2]), decimal)   
        except:
            rounded = 0   

        diff_img = ax[1,0].imshow(np.rot90(array[:,:,2]), vmin=0, vmax=rounded, cmap=cmap)
        fig.colorbar(diff_img, ax=ax[1,0])
        
        diff_img_mask = ax[1,1].imshow(np.rot90(array[:,:,3]), vmin=0, vmax=1, cmap='Reds')
        cbar = fig.colorbar(diff_img_mask, ax=ax[1,1])
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0', '1'])
        
        ax[0,0].set_title('ref')
        ax[0,1].set_title('comp')
        ax[1,0].set_title('difference')
        ax[1,1].set_title(f'masked difference > {threshold}')

        for ii in range(ax.shape[0]*ax.shape[1]):
            ax.flat[ii].axis('off')
            ax.flat[ii].set_facecolor('b')
            fig.set_facecolor('w')
        
        ### [signal, sigma, no_peaks, key]
        fig.suptitle(f'Signal {plot[3][0]} | sigma: {plot[3][1]} | no.peaks: {plot[3][2]} | version: {plot[3][3]}', fontsize=14)
        
        plt.show()
        
        total      = np.sum(plot[2] == True)
        count_ones = np.sum(array[:,:,3] == 1)

        print(f"{string}: match up to the {decimal_match-1}th dec.place ({count_ones}/{total} points deviate by >={threshold}).\n")

def comp_MWF_calculation(cutOutliers=[False, [45,55]], valOutliers=(5,0.00005)):
    
    ''' Estimate deviation between different random states in the same calculation. '''
    
    directory = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\results\100I400P100PSO\T2\GAUSS\Slice12'
    filelist  = glob.glob(f'{directory}\\*.npy')
    ind_MWF, ind_FIT = 5, 6
    PSOdata   = np.load(filelist[0], allow_pickle=True).item()['T2']
    mask      = ~np.isnan(PSOdata[ind_MWF,:,:,-1])
    reference = PSOdata[ind_MWF,:,:,0][mask]
    yy, xx    = cutOutliers[1,0], cutOutliers[1,1]
       
    for i in range(PSOdata.shape[-1])[1:2]:        
        comparison = PSOdata[ind_MWF,:,:,i][mask]  
        title_string = ['T2', 'nn', 'GAUSS', 'version: v_105']
        match_up_arrays(reference, comparison , threshold=10**-2, string=f'{i}', plot=[True,'',mask, title_string])
    
    if cutOutliers == True: 
    
        FITarray     = PSOdata[ind_FIT,yy,xx]
        MWFarray     = PSOdata[ind_MWF,yy,xx]           
        median       = np.median(FITarray)
        stdv         = np.std(MWFarray)
    
        MWFarray     = MWFarray[FITarray<median+stdv*valOutliers[1]] # upper              
        FITarray     = FITarray[FITarray<median+stdv*valOutliers[1]] # upper
        
        plt.plot(FITarray, MWFarray, markersize=2, linestyle='none', marker='o', 
               markeredgewidth=0.5, color='r', markerfacecolor='lightpink',
               label='MWF vs. misfit')
        plt.plot(FITarray[-1], MWFarray[-1], markersize=3, linestyle='none', marker='o', 
               markeredgewidth=0.5, color='k', markerfacecolor='k',
               label='best MWF vs. best misfit')
        
        plt.legend()

def comp_MWF_performance(versions=('v_104', 'v_106'), _sigma=[0.1], _signal=['T2', 'T2S', 'CT2S']):
    
    ''' Tests if similar files created by different PSO version scripts are equal.
        If not, gives back the pre-comma digit on which it is eqaul.
    '''
    
    directory = r'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\results\performance_test'
    filelist  = glob.glob(f'{directory}\\**\\*.npy', recursive=True)
    
    for server in ['MSG', 'MSG2S', 'DS2'][:1]:
        filt_serv = [f for f in filelist if server in Path(f).parts]
        
        for sign in _signal[:]:
            filt_sig = [f for f in filt_serv if f'_{sign}_' in f]
        
            for sig in _sigma[:]:
                filt_sigma = [f for f in filt_sig if str(sig) in os.path.basename(f)]
                
                for no_peaks in ['GAUSS', 'DIRAC'][:]:               
                    filt = [f for f in filt_sigma if no_peaks in os.path.basename(f)]
                    comp_MWF_array(filt, sign, sig, no_peaks, server, versions=versions)     

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
        


# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), tight_layout=True, gridspec_kw={'width_ratios': [1, 1]})
# reds    = plt.cm.Reds(np.linspace(0.3, 0.9, 3))
# ax[0].imshow(rootMWF.data.slice['T2S'][:,:,10])
# for i,pixel in enumerate([[30,30],[45,50],[70,70]]):
#     curve = rootMWF.data.slice['T2'][pixel[0],pixel[1]]
#     ax[1].plot(curve, color=reds[i], label=f'T2 relaxation @ {pixel}')
#     ax[0].scatter(pixel[0],pixel[1], color='red', s=60)

# ax[0].axis('off')
# ax[0].set_facecolor('b')
# fig.set_facecolor('w')

# ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# ax[1].set_yticks([])
# ax[1].set_xticks([])
     
# legend = plt.legend(fontsize=18)
# for text in legend.get_texts():
#     text.set_fontweight('bold')

##############################

# xecution_time_T2S_m1 = timeit.timeit('np.searchsorted(x[0], np.min(mean) - width * np.max(sigma), side='left')',globals=globals(), number=10)
# execution_time_T2_m1 = timeit.timeit('np.matmul(m1_peak,PSOclass.sysMatrix["T2"].T)',globals=globals(), number=10)
# execution_time_T2S_m2 = timeit.timeit('np.matmul(m2_peak,PSOclass.sysMatrix["T2S"].T)',globals=globals(), number=10)
# execution_time_T2_m2 = timeit.timeit('np.matmul(m2_peak,PSOclass.sysMatrix["T2"].T)',globals=globals(), number=10)
# execution_time_T2S_m3 = timeit.timeit('np.matmul(m3_peak,PSOclass.sysMatrix["T2S"].T)',globals=globals(), number=10)
# execution_time_T2_m3 = timeit.timeit('np.matmul(m3_peak,PSOclass.sysMatrix["T2"].T)',globals=globals(), number=10)

# print(f'm1 - T2S: {execution_time_T2S_m1:.5}, T2: {execution_time_T2_m1:.5}')
# print(f'm2 - T2S: {execution_time_T2S_m2:.5}, T2: {execution_time_T2_m2:.5}')
# print(f'm3 - T2S: {execution_time_T2S_m3:.5}, T2: {execution_time_T2_m3:.5}')

# execution_time_add_2 = timeit.timeit('m1_peak+m2_peak',globals=globals(), number=10)
# execution_time_add_3 = timeit.timeit('m1_peak+m2_peak+m3_peak',globals=globals(), number=10)

# print(f'add_2": {execution_time_add_2:.5}, add_3: {execution_time_add_2:.5}')

# execution_time_T2S_2 = timeit.timeit('np.matmul(m1_peak+m2_peak,PSOclass.sysMatrix["T2S"].T)',globals=globals(), number=10)
# execution_time_T2S_3 = timeit.timeit('np.matmul(m1_peak+m2_peak+m3_peak,PSOclass.sysMatrix["T2S"].T)',globals=globals(), number=10)
# execution_time_T2_2 = timeit.timeit('np.matmul(m1_peak+m2_peak,PSOclass.sysMatrix["T2"].T)',globals=globals(), number=10)
# execution_time_T2_3 = timeit.timeit('np.matmul(m1_peak+m2_peak+m3_peak,PSOclass.sysMatrix["T2"].T)',globals=globals(), number=10)

# print(f'T2_2": {execution_time_T2_2:.5}, T2_3: {execution_time_T2_3:.5}')
# print(f'T2S_2": {execution_time_T2S_2:.5}, T2_3: {execution_time_T2S_3:.5}')

# ###
# # plot 3D cube
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# for subject in ['a', 'b', 'c']:
#     directory = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\results\100I400P100PSO\T2\GAUSS'
#     ff  = glob.glob(f'{directory}\\**\\*.npy', recursive=True)
    
#     volume = np.array([np.load(file, allow_pickle=True).item()['T2'] for file in ff[3:22:3]])
    
#     volume[np.isnan(volume)]=0
    
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     spacing = 3
    
#     for i in range(volume.shape[0]):
#         slice_ = volume[i, 5, :, :, -1]
    
#         vmin, vmax = 0.02, 0.28
#         normed = np.clip((slice_ - vmin) / (vmax - vmin), 0, 1)
#         #normed = slice_.astype(float) / np.max(slice_) if np.max(slice_) > 0 else np.zeros_like(slice_)
    
#         # Umgekehrte Graustufen und Transparenz-Map
#         cmap = plt.cm.viridis(normed)
#         cmap[..., -1] = normed  # Alpha = normierter Wert: 0 (schwarz) wird transparent
    
#         x, y = np.meshgrid(np.arange(slice_.shape[1]), np.arange(slice_.shape[0]))
#         z = np.full_like(x, i * spacing)
#         ax.plot_surface(x, y, z, facecolors=cmap, rstride=1, cstride=1, shade=False)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Slice Index')
#     plt.tight_layout()
#     plt.show()
#     savedir = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\results\performance_test'
#     fig.savefig(f'{savedir}\\cube_subject_{subject}.png', dpi=300,format='png')
#     plt.close()
    
# ###
# # plot MWF vs misfit
# for subject in ['a', 'b', 'c']:
#     directory = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_{subject}\results\100I400P100PSO\T2\GAUSS'
#     filelist  = glob.glob(f'{directory}\\**\\*.npy', recursive=True)
#     ff = [f for f in filelist if any(s in f for s in ['Slice04', 'Slice08', 'Slice12', 'Slice16', 'Slice20'])]
    
#     for j, file in enumerate(ff):
#         PSOdata   = np.load(file, allow_pickle=True).item()['T2']
    
#         fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), tight_layout=True)
#         im1 = ax[0].imshow(PSOdata[5,:,:,-1], vmin=0, vmax=0.25)
#         im2 = ax[1].imshow(PSOdata[-1,:,:,-1], vmin=0, vmax=0.005)
#         if j == 0:
#             fig.colorbar(im1, ax=ax[0])
#             fig.colorbar(im2, ax=ax[1])
#         for i in range(2):
#             ax[i].axis('off')
    
#         ID = ['Slice04', 'Slice08', 'Slice12', 'Slice16', 'Slice20'][j]
#         fig.suptitle(f'MWF vs misfit for {ID}.')
    
#         plt.show()
    
#         ID = ['Slice04', 'Slice08', 'Slice12', 'Slice16', 'Slice20'][j]
#         savedir = fr'F:\JIMM2\MWF_invivo\DZNE_Data\24_10_10_c\results\performance_test'
#         fig.savefig(f'{savedir}\\subject_{subject}_{ID}.png', dpi=300,format='png')
#         plt.close()
