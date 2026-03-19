# SPDX-FileCopyrightText: 2026 German Center for Neurodegenerative Diseases - DZNE
# SPDX-FileCopyrightText: 2026 Helmholtz-Zentrum f�r Umweltforschung GmbH - UFZ
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Authors:
#   Tony Stoecker (German Centre for Neurodegenerative Diseases - DZNE)
#   Ségolène Dega (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of the PyMyelinPSO software.
# See the LICENSE file in the project root for full license information.

"""
Modeling of Myelin Water Fraction (MWF) based on on relaxation MRI data (T2,T2star)

Provides three classes:
    - signal_models     defines T2 and T2star signal models (EPG and simple exponential decya)
    - mwf_data          loads T2 and T2star measurement data
    - mwf_analysis      perform MWF analysis on T2 and/or T2star data
"""

import os
import sys
import copy
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter

import nibabel as nib

# check for pymp
pymp_spec = importlib.util.find_spec("pymp")
pymp_found = pymp_spec is not None
if pymp_found:
    import pymp
else:
    print('warning: module pymp not found ... try to continue without parallel processing')

# for windows-based systems --> pymp is not supported
if sys.platform == 'win32':
    pymp_found = False


# check for pyepg
import pyepg


class signal_models():
    """
    Helper class to calculate the system matrix for T2 and T2star signal models
    """

    def __init__(self, signal_type):
        self.signal_type = signal_type  # 'T2', or 'T2S'
        # dictionary of signal model functions to compute system matrix at position (i,j)
        self.mod = {'T2': self.T2_decay, 'T2S': self.T2S_decay, 'CT2S': self.c_T2S_decay}
        self.tgs = []                # time grid signal
        self.tgm = []                # time grid model
        self.fgm = []                # frequency grid model
        self.epg = pyepg.PyEPG()

    def T2S_decay(self, i, j, params=0):
        """simple T2-star decay formula."""
        t = self.tgs[i]
        T2s = self.tgm[j]
        return np.exp(-t / T2s)

    def c_T2S_decay(self, i, j, k, params=0):
        """complex model: T2-star decay with dephasing."""
        # TMP WIP TMP
        t = self.tgs[i]
        T2s = self.tgm[j]
        f = self.fgm[k]
        return np.exp(-t / T2s - 2*np.pi*1j*t*f)

    def T2_decay(self, i, j, params):
        """Multi-Echo Spin-Echo signal amplitudes computed with EPG """
        step  = i
        te    = params['TE']
        beta  = params['beta']
        T1    = params['T1']
        T2    = self.tgm[j]
        tr    = params['TR']
        etl   = params['ETL']
        alpha = params['alpha']

        # init EPG at first echo
        if step == 0:
            self.epg.SetParameters(1.0, T1, T2, te/2)
            self.epg.Equilibrium()
            sat_loops = 4                        # number of T1 saturation loops
            for n in range(sat_loops):
                self.epg.Step(alpha, 0 )           # excitation
                for m in range(etl):             # ETL loop
                    self.epg.Step(beta, 90 )       # refocusing
                    if m < etl-1:
                        self.epg.Step(0, 0 )       # TE delay
                        if n == sat_loops-1:       # after T1-saturation ...
                            return self.epg.GetMagFa(0)  # ... return signal amplitude after first echo! (i==0)
                    else:
                        self.epg.LongDelay(tr-etl*te) # TR delay
        # next step in echo train
        else:
            self.epg.Step(beta, 90)
            self.epg.Step(0, 0)
            return self.epg.GetMagFa(0)

    def system_matrix(self, t_grid_signal, t_grid_model, params, b1scale=0.0):
        """
        compute system matrix for different signal models
        """
        
        m        = len(t_grid_signal)
        n        = len(t_grid_model)
        self.tgs = t_grid_signal
        self.tgm = t_grid_model
        A        = np.zeros((m, n))

        # adjust flip angles, if nonzero b1scale given
        # else, flipangles alpha and beta become zero: eg. in case of no B1 map is given
        if b1scale > 0.0:
            params['alpha'] *= b1scale
            params['beta']  *= b1scale

        for j in range(n):
                for i in range(m):  # inner loop must be over signal time points (EPG of T2_decay)
                    A[i][j] = self.mod[self.signal_type](i, j, params)

        if b1scale > 0.0:  # reset flip angles
            params['alpha'] /= b1scale
            params['alpha']  = np.round(params['alpha'], 2)
            params['beta']  /= b1scale
            params['beta']   = np.round(params['beta'], 2)

        return A

class mwf_data():
    """
    Helper class which loads nifti data and performs some preprocessing for mwf analysis
    It is assumed, that 
    1. all nifti data are in the same voxel space: T2, T2star, and B1 !
    2. the first three dimensions are [x,y,z], the fourth dimension of  T2, T2star is TE (echo times)
    3. B1 is a map of scaling factors, suitable to scale nominal flip angles of the sequences
    """
    def __init__(self, data_dir, KW_T2, KW_T2S, KW_T2SP, KW_B1):
        self.data_dir = data_dir
        self.axis='None'
        self.slice_num=-1
        self.last_slice =-1
        self.last_axis  = "NA"
        self.slice   = {'T2':[],'T2S':[],'CT2S':[],'B1':[]}
        self.ni_t2   = None
        self.ni_t2s  = None
        self.ni_ct2s = None
        self.ni_b1   = None
        self.msk     = None
        files = os.listdir(data_dir)
        #load niftis  by input keywords (use first match)
        if len(KW_T2)>0 and KW_T2 != 'NO_FILE':
            T2  = [s for s in files if KW_T2  in s] 
            if len(T2)>0:   self.ni_t2  = nib.load(os.path.join(data_dir,T2[0]))
            else: print('WARNING: could not find T2 data with keyword %s' % KW_T2)
        if len(KW_T2S)>0 and KW_T2S != 'NO_FILE':
            T2S = [s for s in files if KW_T2S in s] 
            if len(T2S)>0:  self.ni_t2s  = nib.load(os.path.join(data_dir,T2S[0]))
            else: print('WARNING: could not find T2S data with keyword %s' % KW_T2S)
            if len(KW_T2SP)>0 and KW_T2SP != 'NO_FILE':
                T2SP = [s for s in files if KW_T2SP in s] 
                if len(T2SP)>0:  self.ni_ct2s  = nib.load(os.path.join(data_dir,T2SP[0]))
                else: print('WARNING: could not find T2S phase data with keyword %s' % KW_T2SP)
        if len(KW_B1)>0 and KW_B1 != 'NO_FILE':
            B1  = [s for s in files if KW_B1  in s] 
            if len(B1)>0:   self.ni_b1  = nib.load(os.path.join(data_dir,B1[0]))
            else: print('WARNING: could not find B1 data with keyword %s' % KW_B1)

    def __str__(self):
        buf = 'mwf_data      \ndata directory                         : %s' % self.data_dir
        names  = ['T2','T2S','CT2S','B1']
        niftis = [self.ni_t2,self.ni_t2s,self.ni_ct2s,self.ni_b1]
        for i in range(len(niftis)):
            if niftis[i] is None:
                buf += '\n %4s nifti                             : None' % names[i]
            else:
                buf += '\n %4s nifti, shape=%20s : %s' % (names[i],str(np.shape(niftis[i])),os.path.basename(niftis[i].get_filename()))
        buf +=              '\naxis|slice_num                         : %s|%s' % (self.axis, self.slice_num)
        slice_shape  = {'T2':np.shape(self.slice['T2']),'T2S':np.shape(self.slice['T2S']),'CT2S':np.shape(self.slice['CT2S']),'B1':np.shape(self.slice['B1'])}
        buf +=   '\n.slice (data of current slice)         : %s' % str(slice_shape) 
        return buf

    def prep_data(self, axis, slice_num):
        """
        Preprocess 2D image of data for MWF analysis. 
        Input: axis ('X', 'Y', or 'Z') and the slice (number). 
        Results will be stored as member dictionary "slice" with keys "T2", "T2S" (if corresponding data exist)
        """
        
        # nothing to do, slice is already prepped
        if (self.last_axis == axis) and (self.last_slice == slice_num) and self.prepped:
            #print(f'prep_status: {self.prepped}')
            return

        # stores axis and slice number into an object
        self.axis      = axis
        self.slice_num = slice_num
        
        a     =slice(None,None); s=slice(slice_num,slice_num+1) # => x[:,:,slice_num] = x[a,a,s] or x[:,slice_num,:] = x[a,s,a]
        SLICE = {'X':(s,a,a,a),'Y':(a,s,a,a),'Z':(a,a,s,a)} 
        TYPE  = ['T2','T2S','CT2S']
        NIMG  = [self.ni_t2,self.ni_t2s,self.ni_ct2s,self.ni_b1]
        
        # get slice data of T2, T2star (magn), and T2star (phase)
        for i in range(4):
                try:    
                    self.slice[TYPE[i]]  = np.squeeze(NIMG[i].get_fdata()[SLICE[axis]])
                except: 
                    pass # ignore if data not exist
        # complex T2star data: assuming dicom phase range [-4096,-4096] which corresponds to [-pi,pi]
        try:
             factor = 1.0                          # phase values in radiants
             if np.max(self.slice['CT2S'])>1000.0: # assume unprocessed phase from dicom => -4096 < phase < 4096
                    factor = np.pi/4096.0
             self.slice['CT2S'] = self.slice['T2S']*np.exp(1j*self.slice['CT2S']*factor)     
        except:
            pass
        #B1 Map
        SLICE= {'X':(s,a,a),'Y':(a,s,a),'Z':(a,a,s)} 
        self.slice['B1'] = np.squeeze(self.ni_b1.get_fdata()[SLICE[axis]])
        #prep status
        self.last_slice=slice_num
        self.last_axis = axis
        self.prepped   = True

class mwf_analysis():
    
    """
    Class to perform MWF analysis on T2 and/or T2star data.
    """
    def __init__(self, data_dir, KW_B1='NO_FILE', KW_T2='NO_FILE', KW_T2S='NO_FILE', KW_T2SP='NO_FILE',T2S_TE=[], verbose=False):
        """
        Provide input for MWF analysis:
        data_dir:   directory with nifti data (cf. class mwf_data) 
        KW_B1   :   keyword which matches the b1 nifti filename
        KW_T2   :   keyword which matches the T2 nifti filename
        KW_T2S  :   keyword which matches the T2star nifti filename
        T2S_TE  :   name of npy file with 1D-array of TE times (len must match the 4-th dimension of T2S nifti) 
        (TE for T2 will be calculated from sequence parameters, cf. prep_t2_model)
        """
        
        self.data  = mwf_data(data_dir=data_dir, KW_T2=KW_T2, KW_T2S=KW_T2S, KW_T2SP=KW_T2SP, KW_B1=KW_B1)
        self.filter= 0
        self.tsig  = {'T2':[],'T2S':[],'CT2S':[]}       # time points of the signal (TI or TE) - see below
        self.tvnm  = {'T2':'TE','T2S':'TE'} # variable names for signal time points 
        self.dist  = {'T2':[],'T2S':[]}       # example mwf distributions in case of synthetic data
        #nnls result at specific position (from function plot)
        self.nnls  = {'T2':{'mwf':[],'sig':[],'fit':[],'weights':[]},'T2S':{'mwf':[],'sig':[],'fit':[],'weights':[]}}
        #the following members will be defined from functions prep_{t2,t2s}_model
        self.tmod  = {'T2':[],'T2S':[]}       # time points of the model  (T2 or T2* system grid)
        self.sm    = {'T2':[],'T2S':[]}       # system matrices
        self.params= {'T2':None,'T2S':None} # sequence parameters
        self.sig_n_pts = {'T2': [], 'T2S': []}
        self.model_grid_step = {'T2': [], 'T2S': []}

        # the following members will be defined in function mwf_map
        self.mwf = {'T2': [], 'T2S': []}  # result MWF maps
        # result USC maps - the ultra-short-component (?)
        self.usc = {'T2': [], 'T2S': []}
        self.t_thresh_lo = 0
        self.t_thresh_up = 20
        self.nnls_beta = 0

        # get TE times from npy files (TE only for T2S )
        if len(T2S_TE) > 0:
            self.tsig['T2S'] = np.load(os.path.join(data_dir, T2S_TE))

        self.b1_grid = np.zeros((1,))

        self.nprocs  = max(8,os.cpu_count()-2)

    def __str__(self):
        
        '''
        Magical method in python, which is automatically being called, when
        print or str command is being used --> returns argument.
        '''
        
        buf = 'mwf_analysis\n\n .data ->  class %s' % self.data
        buf += '\nin-plane filter width (#voxels)        : %3.2f' % self.filter
        signals = ['T2', 'T2S']
        buf += '\n\n.tsig -> signal time points, i.e. TE values for T2 and T2star signals. Format: (min,max; number)'
        for i in range(len(signals)):
            if len(self.tsig[signals[i]]) == 0:
                buf += '\n tsig[\'%s\'] : None' % signals[i]
            else:
                t = self.tsig[signals[i]]
                buf += '\n tsig[\'%s\'] : (%5.3f,%5.3f; %d)' % (
                    signals[i], min(t), max(t), len(t))
        buf += '\n\n.tmod -> model time points for T2 and T2star. Format: (min,max; number)'
        for i in range(len(signals)):
            if len(self.tmod[signals[i]]) == 0:
                buf += '\n tmod[\'%s\'] : None' % signals[i]
            else:
                t = self.tmod[signals[i]]
                buf += '\n tmod[\'%s\'] : (%5.3f,%5.3f; %d)' % (
                    signals[i], min(t), max(t), len(t))
        buf += '\n\n.params -> sequence parameters'
        for i in range(len(signals)):
            if self.params[signals[i]] is None:
                buf += '\n params[\'%s\'] : None' % signals[i]
            else:
                buf += '\n params[\'%s\'] : %s' % (
                    signals[i], self.params[signals[i]])
        sm_shape =  {'T2': np.shape(self.sm['T2']), 'T2S': np.shape(self.sm['T2S'])}
        mwf_shape = {'T2': np.shape(self.mwf['T2']), 'T2S': np.shape(self.mwf['T2S'])}
        buf += '\n\n.sm  -> system matrix : %s' % str(sm_shape)
        buf += '\n.mwf -> MWF maps      : %s' % str(mwf_shape)
        #buf +=   '\nfit params            : nnls_beta=%5.4f, MWF thresh (lower,upper) = %d,%d  ' % (self.nnls_beta, self.t_thresh_lo, self.t_thresh_up)
        return buf

    def prep_data(self, axis, slice_num, signal_type, thresh=10.0, 
                  filter=0.0, N_b1_grid=50, verbose=False):
        """
        Prepare single slice to perform MWF analysis
        Input
        axis        :   'X', 'Y', or 'Z'
        slice_num   :   slice number   
        signal_type :   'T2' or 'T2S'
        thresh      :   percentage threshold for masking (default: 10)
        filter      :   voxel-width of in-plane image filter for smoothing. default = 0 (no filter)
        N_b1_grid   :   number of points for the B1 grid of the system cube (one system matrix for each point) 
        """
              
        # calls data and writes it into an array_like
        self.data.prep_data(axis.upper(), slice_num)
        data = self.data.slice[signal_type]
        tax  = self.tsig[signal_type]    # time points of the signal (TI or TE)
        
        # define additional parameters, if signal is complex
        if signal_type == 'CT2S':
            cmplx_signal = True
            signal_type  = 'T2S'
        else:
            cmplx_signal = False
            
        # avoid error message if T2 time axis is missing for plot (which is the case if the T2 model is not yet prepared)
        if verbose and len(tax) == 0 and signal_type == 'T2':
            tax = range(len(data[0, 0, :]))
            XL = 'echo #'
        else:
            XL = self.tvnm[signal_type]+' ms'

        # optional in-plane data smoothing to increase SNR of time signals
        # 1D smoothing filter is being applied over the x and y-axis
        if filter > 0.0:
            self.filter = filter
            if cmplx_signal: # filter real and imaginry parts independently (avoid to filter the phase)
                filt_real = gaussian_filter(np.real(data),(filter, filter, 0))
                filt_imag = gaussian_filter(np.imag(data),(filter, filter, 0))
                self.data.slice['CT2S'] = filt_real+1j*filt_imag                
            data                         = gaussian_filter(np.abs(data), (filter, filter, 0))
            self.data.slice[signal_type] = data
            self.filter                  = filter

        # compute rough mask from first image of current contrast
        # !!! better to use segmentation (additional script) for future steps
        msk = np.abs(data[:, :, 0])
        mx  = np.max(msk)
        msk[(thresh*msk/mx) < 1.0] = 0.0
        msk[(thresh*msk/mx) > 1.0] = 1.0
        self.data.msk = msk

        # If b1-map has large values, then assume relative B1 from 3DREAM, where 1000 corresponds to the nominal FA
        if (np.max(self.data.slice['B1']) > 100.0):
            # print(f'max: {np.max(self.data.slice["B1"])}')
            self.data.slice['B1'] = self.data.slice['B1']/1000
            
        # range of B1 scalings in mask and also for calculating b1_inhomogenity grid
        b1_msk = self.data.slice['B1']*msk
        b1_msk[b1_msk == 0] = np.nan
        b1_min = np.nanmin(b1_msk)
        b1_max = np.nanmax(b1_msk)

        # Calculate b1_grid only for T2 to avoid changes from T2S values in JI applications
        # → prevents unintended modifications of the T2 matrix through inheritance        
        if signal_type == 'T2':
            self.b1_grid = np.linspace(b1_min, b1_max, N_b1_grid)        

        # print actual status and plot actual data
        if verbose:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 30))
            im1 = ax1.imshow(np.abs(data[:, :, 0]), cmap='gray')
            im2 = ax2.imshow(msk, cmap='gray')
            im3 = ax3.imshow(self.data.slice['B1'], cmap='gray')
            ax1.axis('off'), ax1.set_title('$|$%s$_{n=0}|$' % (signal_type))
            ax2.axis('off'), ax2.set_title('mask')
            ax3.axis('off'), ax3.set_title('B1')
            (x_1, y_1) = np.shape(msk)
            x = x_1//2
            y = y_1//2  # center voxel
            ax1.scatter(y, x, s=5, c='red', marker='o')
            ax4.plot(tax, data[x, y, :]/np.abs(data[x, y, 0]))
            ax4.set_title('%s signal' % signal_type)
            ax4.set_xlabel('%s' % XL)
            ax4.set_aspect(1/ax4.get_data_ratio())
            plt.show()
            print('axis, slice number   = (%s, %d)' % (axis, slice_num))
            print('min , max FA scaling = (%6.4f, %6.4f)' % (b1_min, b1_max))

    def prep_synthetic_data(self, data_path, axis, slice_num, signal_type, mwf_thresh=0.025, 
                            SNR=[500.0, 0], seed=5, nprocs=8, dmean=[], dstdv=[], 
                            x=0, y=0, complex_T2S=False, phases=[0,0,0], verbose=True):
        
        # default values: mean and standard deviation of 2 Gaussian peaks for synthetic signals
        dist_mean      = {'T2': [20, 80], 'T2S': [10, 40]}
        dist_stdv      = {'T2': [10, 10], 'T2S':  [8, 8]}
        self.synthetic = True
        
        # keyword values: mean and standard deviation of Gaussian peaks for synthetic signals
        if len(dmean) == 2:
            dist_mean[signal_type] = dmean
        if len(dstdv) == 2:
            dist_stdv[signal_type] = dstdv
        
        # call mwf values from atlas into a numpy array
        # __file__ is the absolute path of the actual python script
        self.data.axis      = axis
        self.data.slice_num = slice_num

        mwf_mean = nib.load(data_path)
        
        # slicing => x[:,:,slice_num] = x[a,a,s] or x[:,slice_num,:] = x[a,s,a]
        # np.squeeze helps to reduce unnecessary dimensions
        a        = slice(None, None)
        s        = slice(slice_num, slice_num+1)
        SLICE    = {'X': (s, a, a), 'Y': (a, s, a), 'Z': (a, a, s)}        
        mwf      = np.squeeze(mwf_mean.get_fdata()[SLICE[axis]])
        (n1, n2) = mwf.shape
        
        # misuse the b1-map entry to keep a copy of the mwf slice
        self.data.slice['B1'] = np.copy(mwf)
        
        # build a rough mask dependent on a threshold of MWF values
        msk = mwf.copy()
        msk[msk < mwf_thresh] = 0.0
        msk[msk > mwf_thresh] = 1.0
        self.data.msk = msk
        
        # relaxation distribution with two components: MW=mwf , AEW=1-mwf
        # calculates the gaussians
        
        def distribution(mwf):
            tm = self.tmod[signal_type]
            m = dist_mean[signal_type]
            s = dist_stdv[signal_type]
            fast_component = mwf*np.exp(-(m[0]-tm)**2/(2*s[0]**2))
            slow_component = (1-mwf)*np.exp(-(m[1]-tm)**2/(2*s[1]**2))
            return (fast_component, slow_component)
        
        # store example distribution for mwf[x,y] in the center of the array/image
        if x == 0:
            x = n1//2
        if y == 0:
            y = n2//2
        self.dist[signal_type] = distribution(mwf[x, y])
        
        # print status update
        if verbose:
            print('mwf(%d,%d)=%3.2f \n' % (x, y, mwf[x, y]))

            if pymp_found:
                with pymp.Parallel(1) as p:
                    with p.lock:
                        if verbose:
                            p.print('computing ', signal_type, ' synthetic signals: axis|slice_num =',
                                    self.data.axis, '|', self.data.slice_num)
            else:
                print('computing ', signal_type, ' synthetic signals (without parallel computing): axis|slice_num =',
                      self.data.axis, '|', self.data.slice_num)

            
        # compute synthetic signals         
        if pymp_found:
            data = pymp.shared.array(
                (n1, n2, len(self.tsig[signal_type])), dtype='double')
            progress = pymp.shared.array((1,), dtype='uint8')
            data_complex = np.zeros((n1, n2, len(self.tsig[signal_type])))
        else:
            data = np.zeros((n1, n2, len(self.tsig[signal_type])))
            data_complex = np.zeros((n1, n2, len(self.tsig[signal_type])))    
            progress = np.zeros((1,))

        def inner_loop(i):
            for j in range(n2):
                if msk[i, j] > 0:
                    if seed > 0.0 and SNR[1] == 0:
                        print('SNR correlated')
                        np.random.seed(seed)
                        
                    fast_component, slow_component = distribution(mwf[i, j])
                    A = self.sm[signal_type]
                    
                    if complex_T2S and signal_type=='T2S':
                        
                        fast_decay = self.integrate_distribution(
                            A[:, :, 0], fast_component, 'T2S')
                        
                        fast_decay = np.multiply(
                            fast_decay, np.exp(-2j*np.pi*phases[0]*self.tsig['T2S']/1000),
                            dtype=np.complex128)
                        
                        slow_decay = self.integrate_distribution(
                            A[:, :, 0], slow_component, 'T2S')
                        
                        slow_decay = np.multiply(
                            slow_decay, np.exp(-2j*np.pi*phases[1]*self.tsig['T2S']/1000),
                            dtype=np.complex128)
                        

                        signal = np.add(fast_decay, slow_decay)
                        signal *= np.exp(-1j*phases[2])
                        signal_real = np.real(signal)
                        signal_imag = np.imag(signal)
                        
                        if SNR[0] != 0:
                            signal_real += np.random.normal(loc=0, scale=np.abs(
                                signal_real[0]) / SNR[0], size=np.shape(signal_real))
                            signal_imag += np.random.normal(loc=0, scale=np.abs(
                                signal_imag[0]) / SNR[0], size=np.shape(signal_imag))

                        signal = signal_real + signal_imag*1j
                        data[i, j, :] = np.abs(signal)
                        data_complex[i, j, :] = np.angle(signal)
                    else:
                        d = np.add(fast_component, slow_component)
                        signal = np.dot(A[:, :, 0], d)
                        
                        if SNR[0] != 0:
                            signal += np.random.normal(loc=0, scale=np.abs(
                                signal[0]) / SNR[0], size=np.shape(signal))
                            
                        data[i, j, :] = np.abs(signal)

        if pymp_found:
            with pymp.Parallel(nprocs) as p:
                for i in p.range(0, n1):
                    with p.lock:
                        progress[0] += 1
                    if (np.mod(progress[0], 10) == 0):
                        if verbose:
                            p.print('progress: ', progress[0], ' / ', n1)
                    inner_loop(i)
        else:
            for i in range(n1):
                progress[0] += 1
                if (np.mod(progress[0], 10) == 0):
                    if verbose:
                        print('progress: ', progress[0], ' / ', n1)
                inner_loop(i)

        self.data.slice[signal_type] = data
        
        if complex_T2S and signal_type=='T2S':
            # data                    - magnitude
            # data_complex            - phase of the signal
            # np.exp(1j*data_complex) - 
            # multiplication          - signal in polar coordinates
            self.data.slice['CT2S'] = data*np.exp(1j*data_complex)
        
        return self.data.msk[...,np.newaxis]
    
    def prep_t2_model(self, te=6.6, tr=900, etl=24, alpha=70, beta=180, 
                      T2min=1, T2max=300, nT2=300, T1=1000, verbose=True):
        
        '''
        Preparation of T2 model and computation of the system cube.
        '''
        
        # CPMG signal model and sequence parameters
        params            = {'TE': te, 'TR': tr, 'ETL': etl, 
                             'alpha': alpha, 'beta': beta, 'T1': T1}
        self.params['T2'] = params
        
        # signal time points for T2 defined by sequence parameters TE and ETL
        self.tsig['T2']            = np.linspace(te, te*etl, etl)
        self.sig_n_pts['T2']       = len(self.tsig['T2'])
        self.tmod['T2']            = np.linspace(T2min, T2max, nT2)
        self.model_grid_step['T2'] = (T2max - T2min) / nT2
        
        # compute system matrix
        self.system_cube(signal_type='T2', params=params, verbose=verbose)

    def prep_t2s_model(self, T2Smin=1, T2Smax=200, nT2S=200, verbose=True):
        """
        prepare T2star model and compute system cube
        """
        # simple exponential  signal decay model
        # no seq params needed for T2* /  alpha, beta for compatibility with T2 model
        params = {"alpha": 0.0, "beta": 0.0}
        
        self.params['T2S'] = params
        self.sig_n_pts['T2S'] = len(self.tsig['T2S'])
        self.tmod['T2S'] = np.linspace(T2Smin, T2Smax, nT2S)
        self.model_grid_step['T2S'] = (T2Smax - T2Smin) / nT2S
        
        # compute system matrix
        self.system_cube(signal_type='T2S', params=params, verbose=verbose)

    def mwf_map(self, signal_type, mwf_thresh=(50, 500), nnls_beta=0.0, DS=1, verbose=True):
        """
        compute MWF map with NNLS
        """
        if len(self.sm[signal_type]) == 0:
            print('abort - %s system matrix missing => run prep_%s_model() first!' %
                  (signal_type, signal_type.lower()))
            return

        self.mwf[signal_type], self.usc[signal_type] = self.mwf_map_nnls(signal_type=signal_type, mwf_thresh=mwf_thresh, nnls_beta=nnls_beta, DS=DS, verbose=verbose)

    def system_cube(self, signal_type, params, verbose):
        '''
        System matrix for T2 or T2* signal. 
        In case of T2, a system cube is being created
        --> based on all b1 scaling factors (called by prep_{t2,t2s}_model)

        '''
        
        # print status
        if verbose:
            print('computing ', signal_type, ' system matrix')
            
        # ignore B1-map (flip-angle correction) for T2-star model (the system cube is in fact a system matrix)
        # flip angle correction should also be ignored for inverting MWF Atlas
        if signal_type == 'T2S':
            backup = self.b1_grid.copy()
            self.b1_grid = np.zeros((1,))

        # signal model  - signal model function to compute system matrix at position (i,j)
        # t_grid_model  - system grid over 1000 steps
        # t_grid_signal - timesteps for decay in ms and in 6.6 ms intervalls
        signal_model  = signal_models(signal_type=signal_type)
        t_grid_model  = self.tmod[signal_type]
        t_grid_signal = self.tsig[signal_type]

        # build an empty matrix array with dimensions [sig_len, grid_size, b1_size]
        if pymp_found:
            A = pymp.shared.array((len(t_grid_signal), len(t_grid_model), len(self.b1_grid)), dtype='double')
            progress = pymp.shared.array((1,), dtype='uint8')
        else:
            A = np.zeros((len(t_grid_signal), len(t_grid_model), len(self.b1_grid)))
            progress = np.zeros((1,))

        # fill the system matrix for each voxel
        if pymp_found and len(self.b1_grid)>1:
            with pymp.Parallel(self.nprocs) as p:
                for i in p.range(0, len(self.b1_grid)):
                    with p.lock:
                        progress[0] += 1
                    if (np.mod(progress[0], 10) == 0):
                        if verbose:
                            p.print('progress: ', progress[0], ' / ', n1) # where did we loose n1 ??
                    A[:, :, i] = signal_model.system_matrix(
                    t_grid_signal=t_grid_signal, t_grid_model=t_grid_model, params=params, b1scale=self.b1_grid[i])
        else:
            for i in range(len(self.b1_grid)):
                progress[0] += 1
                if (np.mod(progress[0], 10) == 0):
                    if verbose:
                        print('progress: ', progress[0], ' / ', len(self.b1_grid))
                A[:, :, i] = signal_model.system_matrix(
                    t_grid_signal=t_grid_signal, t_grid_model=t_grid_model, params=params, b1scale=self.b1_grid[i])

        if signal_type == 'T2S':
            self.b1_grid = backup.copy()

        self.sm[signal_type] = A

    def compute_gaussian(self, x, sigma, mean, scale):
        """ Compute a Gaussian function along the x vector
        sigma : standard deviation of the Gaussian
        mean  : mean of the Gaussian
        scale : scale applied to the guassian (integral)
        """
        gaussian_scale = scale / (np.sqrt(2*np.pi)*sigma)
        gaussian_exp = np.exp(-np.square(x - mean) / (2 * sigma ** 2))
        return gaussian_scale*gaussian_exp

    def compute_decay_curve(self, model, matrix, signal_type,
                        model_component="Gaussian",
                        T2S_complex=False,
                        T2S_frequency_shift=False,
                        return_distribution=False):
        """
        Compute T2 or T2s decay curve from model parameters
        model: model parameters(slow and fast components mean and standard
                                deviation, and fast component integral)
        signal_type : T2 or T2s
        """
        if model_component == "Gaussian":
            fast_component = self.compute_gaussian(
                self.tmod[signal_type], model[1], model[0],
                model[4]*model[5]/(1-model[5]))
            slow_component = self.compute_gaussian(
                self.tmod[signal_type], model[3], model[2], model[4])
    
        if model_component == "3Diracs":
            index_components = self.find_component_indexes(signal_type, model)
            fast_component, slow_component = self.set_component_amplitudes(
                signal_type, model, index_components)
    
        T_distribution = np.add(fast_component, slow_component)
        if T2S_complex:
            fast_decay = self.integrate_distribution(
                matrix, fast_component, signal_type)
            fast_decay = np.multiply(
                fast_decay, np.exp(-2j*np.pi*model[6]*self.tsig['T2S']/1000),
                dtype=np.complex128)
            slow_decay = self.integrate_distribution(
                matrix, slow_component, signal_type)
            slow_decay = np.multiply(
                slow_decay, np.exp(-2j*np.pi*model[7]*self.tsig['T2S']/1000),
                dtype=np.complex128)
            decay_curve = np.add(fast_decay, slow_decay)    # magnitude
            decay_curve *= np.exp(-1j*model[8])             # complex signal (*phi)
            decay_curve = np.concatenate(
                (np.abs(decay_curve), np.unwrap(np.angle(decay_curve))))
        elif T2S_frequency_shift:
            fast_decay = self.integrate_distribution(
                matrix, fast_component, signal_type)
            fast_decay = np.multiply(
                fast_decay, np.exp(-2j*np.pi*model[6]*self.tsig['T2S']/1000),
                dtype=np.complex128)
            slow_decay = self.integrate_distribution(
                matrix, slow_component, signal_type)
            decay_curve = np.abs(np.add(fast_decay, slow_decay,
                                        dtype=np.complex128))
        else:
            decay_curve = self.integrate_distribution(
                matrix, T_distribution, signal_type)
    
        if return_distribution:
            return decay_curve, T_distribution
        else:
            return decay_curve
    
    def find_component_indexes(self, signal_type, model):
        index_components = [0, 0, 0]
        index_components[0] = np.abs(
            self.tmod[signal_type] - model[0]).argmin()
        index_components[1] = np.abs(
            self.tmod[signal_type] - model[1]).argmin()
        index_components[2] = np.abs(
            self.tmod[signal_type] - model[2]).argmin()
        return index_components
    
    def set_component_amplitudes(self, signal_type, model, index_components):
        fast_component = np.zeros(len(self.tmod[signal_type]))
        slow_component1 = np.zeros(len(self.tmod[signal_type]))
        slow_component2 = np.zeros(len(self.tmod[signal_type]))
        fast_component[index_components[0]] = (model[3]+model[4]) * \
            model[5]/(1-model[5])
        slow_component1[index_components[1]] = model[3]
        slow_component2[index_components[2]] = model[4]
        slow_component = np.add(slow_component1, slow_component2)
        return fast_component, slow_component
    
    def integrate_distribution(self, matrix, distribution, signal_type):
        decay_curve = np.matmul(matrix, distribution)
        decay_curve *= self.model_grid_step[signal_type]
        return decay_curve


    def nnls_reg_simple(self, A, b, beta, p=0.0):
        """
        solving min_x ||A*x - b||^2 + beta^2||x-p||^2 by letting C=[A, beta*I] and d=[b, beta*p], 
        then solving min_x ||C*x - d||^2 with scipy.optimize.nnls. beta is the regularization 
        parameter and p is a constant prior for the solution, which must be either scalar or 
        of same dimension as solution x.
        """
        if beta == 0.0:
            return nnls(A, b)
        I = np.identity(np.size(A, axis=1))
        C = np.concatenate((A, beta*I), axis=0)
        if np.isscalar(p):
            p = p*np.ones(np.size(A, axis=1))
        x = np.concatenate((b, beta*p), axis=0)
        return nnls(C, x)

    def mwf_nnls(self, signal, T_grid, A, beta=0.0, T_thresh_lo=0.0, T_thresh_up=20.0):
        """
        estimate the MWF using NNLS.
        Input (signal, T_grid, A, beta, T_thresh_lo, T_thresh_up)
        signal      : shape (n,) MR signal (T2 or T2*)
        T_grid      : shape (m,) time grid of the model 
        A           : shape (n,m) system matrix 
        beta        : NNLS regularization parameter (default: 0.0)
        T_thresh_lo : lower threshold for MWF component (default: 0.0)
        T_thresh_up : upper threshold for MWF component (default: 20.0)
        Returns
        (mwf, usc, weights)
        mwf (scalar): myelin water fraction (T_thresh_lo < T <T_thresh_up)
        usc (scalar): ultra-short component (T < T_thresh_lo) 
        weights     : shape (m,) relaxation distribution estimate
        """
        weights, chi2 = self.nnls_reg_simple(A, signal, beta)

        w_ultrashort = weights[T_grid < T_thresh_lo]
        w_fast_comp = np.setdiff1d(weights[T_grid < T_thresh_up], w_ultrashort)
        w_slow_comp = weights[T_grid > T_thresh_up]

        if w_ultrashort.size == 0:
            w_ultrashort = 0.0
        if w_fast_comp.size == 0:
            w_fast_comp = 0.0
        if w_slow_comp.size == 0:
            w_slow_comp = 0.0

        UC = np.sum(w_ultrashort)
        FC = np.sum(w_fast_comp)
        SC = np.sum(w_slow_comp)
        if SC == 0:
            SC = max(np.finfo(float).eps, np.sum(weights))

        return (FC / (UC+FC+SC), UC / (UC+FC+SC), weights)

    def mwf_map_nnls(self, signal_type, mwf_thresh, nnls_beta, DS, verbose):
        # calculate MWF in one slice for T2 or T2* using NNLS (called by mwf_map)

        # ignore B1-map (flip-angle correction) for T2-star model
        if signal_type == 'T2S':
            backup = self.b1_grid.copy()
            self.b1_grid = np.zeros((1,))

        data = self.data.slice[signal_type]
        t_grid = self.tmod[signal_type]
        A = self.sm[signal_type]

        if verbose:
            if pymp_found:
                with pymp.Parallel(1) as p:
                    with p.lock:
                        if verbose:
                            p.print('computing ', signal_type, ' mwf map: axis|slice_num =',
                                    self.data.axis, '|', self.data.slice_num)
            else:
                print('computing ', signal_type, ' mwf map (without parallel computing): axis|slice_num =',
                      self.data.axis, '|', self.data.slice_num)

        (x_1, y_1, z_1) = np.shape(data)
        n1 = int(x_1/DS)
        n2 = int(y_1/DS)
        # parameter for NNLS MWF fit
        t_thresh_lo = mwf_thresh[0]  # lower bound for MW fraction
        t_thresh_up = mwf_thresh[1]  # upper bound for MW fraction
        # remember current values for plot function (which runs NNLS in a single voxel)
        self.t_thresh_lo = t_thresh_lo
        self.t_thresh_up = t_thresh_up
        self.nnls_beta = nnls_beta

        # NNLS estimate for all voxels. First dimension in parallel loop using pymp, if available
        if pymp_found:
            mwf = pymp.shared.array((n1, n2), dtype='double')
            usc = pymp.shared.array((n1, n2), dtype='double')
            progress = pymp.shared.array((1,), dtype='uint8')
            voxels = pymp.shared.array((1,), dtype='uint8')
        else:
            mwf = np.zeros((n1, n2))
            usc = np.zeros((n1, n2))
            progress = np.zeros((1,))
            voxels = np.zeros((1,))

        def inner_loop(i):
            for j in range(n2):
                if self.data.msk[DS*i, DS*j] == 1:
                    voxels[0] += 1  # count number of voxels in mask
                    signal = data[DS*i, DS*j, :]
                    n_fs = np.argmin(
                        np.abs(self.b1_grid-self.data.slice['B1'][DS*i, DS*j]))
                    mwf[i, j], usc[i, j], _ = self.mwf_nnls(signal, t_grid, A[:, :, n_fs], nnls_beta, t_thresh_lo, t_thresh_up)

        if pymp_found:
            with pymp.Parallel(self.nprocs) as p:
                for i in p.range(0, n1):
                    with p.lock:
                        progress[0] += 1
                    if (np.mod(progress[0], 10) == 0):
                        if verbose:
                            p.print('progress: ', progress[0], ' / ', n1)
                    inner_loop(i)
        else:
            for i in range(n1):
                progress[0] += 1
                if (np.mod(progress[0], 10) == 0):
                    if verbose:
                        print('progress: ', progress[0], ' / ', n1)
                inner_loop(i)

        if signal_type == 'T2S':
            self.b1_grid = backup.copy()

        return mwf, usc

    def plot(self, signal_type, component='SUM', vmax=0.5, x=0, y=0, alpha=0.7, DS=1, save=False, prefix='mwf_plot', verbose=True):
        """
        plot MWF map and signal / fit in a single voxel
        """
        # what to plot? MWF, USC, or SUM (i.e. sum of both components)
        if component == 'MWF':
            img = self.mwf[signal_type].copy()
        elif component == 'USC':
            img = self.usc[signal_type].copy()
        elif component == 'SUM':
            img = self.mwf[signal_type]+self.usc[signal_type]
        else:
            print('component must be one of MWF, USC, SUM')
            return

        # mask out values > vmax
        img[img > vmax] = 0.0
        img = np.ma.masked_where(img == 0.0, img)
        cmap = copy.copy(plt.cm.get_cmap('viridis'))
        cmap.set_bad(color='black')

        # background image for plot
        bg = np.abs(self.data.slice[signal_type][::DS, ::DS, 0])

        # ignore B1-map (flip-angle correction) for T2-star model
        if signal_type == 'T2S':
            backup = self.b1_grid.copy()
            self.b1_grid = np.zeros((1,))

        # calculate NNLS in single voxel
        (n1, n2) = np.shape(img)
        if x == 0:
            x = DS*n1//2
        if y == 0:
            y = DS*n2//2
        x = x//DS
        y = y//DS  # voxel position
        signal = self.data.slice[signal_type][DS*x, DS*y, :]
        n_fs = np.argmin(
            np.abs(self.b1_grid-self.data.slice['B1'][DS*x, DS*y]))
        A = self.sm[signal_type][:, :, n_fs]
        mwf_xy, dmc_xy, weights = self.mwf_nnls(signal, self.tmod[signal_type], A,
                                           self.nnls_beta, self.t_thresh_lo, self.t_thresh_up)
        fit = np.dot(A, weights)
        # normalize
        signal = signal / np.max(np.abs(signal))
        fit = fit / np.max(np.abs(fit))

        if signal_type == 'T2S':
            self.b1_grid = backup.copy()

        # store signal and fit for later use
        self.nnls[signal_type]['mwf'] = mwf_xy
        self.nnls[signal_type]['sig'] = signal
        self.nnls[signal_type]['fit'] = fit
        self.nnls[signal_type]['weights'] = weights

        # stop here, if no plot was requested
        if verbose == False:
            return

        # plot map as well as signal/fit and nnls results in single voxel
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
        ax1.imshow(bg, cmap='gray')
        im = ax1.imshow(img, cmap=cmap, alpha=alpha)
        ax1.axis('off')
        ax1.set_title('%s(%d,%d) = %5.2f' % (component, x, y, img[x, y]))
        ax1.scatter(y, x, s=5, c='red', marker='o')
        fig.colorbar(im, ax=ax1)

        ax2.plot(self.tsig[signal_type], signal, 'o-', label='signal')
        ax2.plot(self.tsig[signal_type], fit, '-', label='fit')
        ax2.set_xlabel('%s [ms]' % self.tvnm[signal_type])
        ax2.set_title('%s signal (norm.)' % (signal_type))
        ax2.legend()

        ax3.plot(self.tmod[signal_type], weights)
        ax3.plot([self.t_thresh_lo, self.t_thresh_lo], [
                 0, np.max(weights)], 'k', linestyle=':')
        ax3.plot([self.t_thresh_up, self.t_thresh_up], [
                 0, np.max(weights)], 'k', linestyle=':')
        ax3.set_xlabel('%s [ms]' % (signal_type))
        ax3.set_title('MWF=%4.2f , USC=%4.2f' % (mwf_xy, dmc_xy))

        fig.set_facecolor('w')
        if save:
            plt.savefig('%s_%s_%s%d_beta%5.4f.png' % (
                prefix, signal_type, self.data.axis, self.data.slice_num, self.nnls_beta))

        plt.show()