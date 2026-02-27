# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# Authors:
#   Martin Kobe (Helmholtz Centre for Environmental Research - UFZ)
#   Ségolène Dega (Helmholtz Centre for Environmental Research - UFZ)
#
# This file is part of the PyMRI_PSO software.
# See the LICENSE file in the project root for full license information.

"""
Helper functions supporting efficient PSO execution on in-vivo and atlas MRI data.
"""

import sys, yaml, json, time, matplotlib
from   numba               import njit
import numpy               as     np
import matplotlib.pyplot   as     plt
from   matplotlib.widgets  import RectangleSelector

###############################################################################
# define a binary mask for an image: 0 is masked, 1 is visible
# --> click on the pixel you want to reduce --> it turns from 1 to 0

def mask_shape(binary_array: np.array, comparison_array: np.array, layer=0,
               transparency=0.5, selection_mode='pix'):
    
    ''' mask a binary array by toggling
    
    args:
        binary_array: 2 dimensional binary input array
        comparison_array: 2 dimensional comparison array
        transparency: alpha level within grafics
        selection_mode: ['pix', 'rec'] - for pixel or rectangle
    
    returns:
        updated copy of binary array
    '''

    mask_change = np.copy(binary_array)
    
    if selection_mode == 'pix':
        mask_change = mask_shape_pixel(binary_array=mask_change, comparison_array=comparison_array, 
                                       i=layer, transparency=transparency)  
    
    if selection_mode == 'rec':
        mask_change = mask_shape_rectangle(binary_array=mask_change, comparison_array=comparison_array, 
                                           i=layer, transparency=transparency)
        
    if selection_mode == 'both':
        mask_change = mask_shape_combo(binary_array=mask_change, comparison_array=comparison_array, 
                                       i=layer, transparency=transparency)  
    
    return mask_change


def mask_shape_combo(binary_array: np.array, comparison_array: np.array, i=0, transparency=0.5):
    
    matplotlib.use('Qt5Agg')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(comparison_array)
    img = ax[0].imshow(binary_array, cmap='gray', alpha=transparency, vmin=0, vmax=1)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title(f"Slide {str(i).zfill(2)} - Left Click: toggle pixel | drag: clear rectangle")

    ax[1].imshow(comparison_array)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("Comparison Only")

    # Toggle einzelner Pixel per Klick
    def onclick(event):
        if event.inaxes != ax[0] or event.button != 1 or event.dblclick:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not rectangle_selector.active:
            return
        # toggle pixel
        x, y = int(event.xdata), int(event.ydata)
        binary_array[y, x] = 1 - binary_array[y, x]
        img.set_data(binary_array)
        plt.draw()

    # Rechteck löscht Maske (setzt nur auf 0)
    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        binary_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1] = 0
        img.set_data(binary_array)
        plt.draw()

    rectangle_selector = RectangleSelector(
        ax[0], onselect, useblit=True, button=[1],
        interactive=False, drag_from_anywhere=True,
        props=dict(facecolor='red', edgecolor='black', alpha=0.3, fill=True))

    fig.canvas.mpl_connect('button_press_event', onclick)

    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

    return binary_array

def mask_shape_pixel(binary_array: np.array, comparison_array: np.array, i=0, transparency=0.5):

    # tell matplotplib to use an extra window for drawing the graphic    
    matplotlib.use('Qt5Agg')

    # function for selection an area by rectangle    
    def onclick(event):
        x, y = int(event.xdata), int(event.ydata)
        if x is not None and y is not None:
            binary_array[y, x] = 1 - binary_array[y, x]
            img.set_data(binary_array)
            plt.draw()

    # calling a figure and draw the mask transparent above a comparison array  
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    ax[0].imshow(comparison_array)   
    img  = ax[0].imshow(binary_array, cmap='gray', alpha=0.5, vmin=0, vmax=1)
    
    ax[0].set_xticks([]); ax[0].set_yticks([])    
    ax[0].set_title(f"Slide {str(i).zfill(2)} - click on a pixel to toggle it")
    
    ax[1].imshow(comparison_array)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("Comparison Only")

    fig.canvas.mpl_connect('button_press_event', onclick)
    
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
        
    return binary_array
        
def mask_shape_rectangle(binary_array: np.array, comparison_array: np.array, i=0, transparency=0.5):
    
    # tell matplotplib to use an extra window for drawing the graphic
    matplotlib.use('Qt5Agg')
    
    # function for selection an area by rectangle
    def onselect(eclick, erelease):
        # Start- und Endkoordinaten des Rechtecks
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Bereich auswählen und auf 0 setzen
        # binary_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x2, x2)+1] = 0
        binary_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1] = \
                    1 - binary_array[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1]
        
        # Array live aktualisieren
        img.set_data(binary_array)
        plt.draw()
    
    # calling a figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(comparison_array)
    img = ax[0].imshow(binary_array, cmap='gray', alpha=transparency, vmin=0, vmax=1)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title(f"Slide {str(i).zfill(2)} - Toggle Rectangle Mask")
    
    ax[1].imshow(comparison_array)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("Comparison Only")
        
    # activate the rectangle selector
    rectangle_selector = RectangleSelector(ax[0], onselect, useblit=True, button=[1], interactive=False, 
                                           props=dict(facecolor='red', edgecolor='black', alpha=0.3, fill=True))
    rectangle_selector.set_active(True)
    
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
    
    return binary_array    

###############################################################################
### conversion functions ######################################################

def conv_npy2mat(file_npy: str):
    
    from   pathlib  import Path
    import numpy    as     np
    from   scipy.io import savemat

    file = Path(file_npy)
    arr  = np.load(file)
    savemat(file.with_suffix('.mat'), {"data": arr})

def conv_npy2nii(file_npy: str):
    
    from pathlib import Path
    import numpy as np
    import nibabel as nib

    file = Path(file_npy)
    arr  = np.load(file)
    nii  = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nii, file.with_suffix(".nii"))

###############################################################################
### time log ##################################################################
    
def log(start_time, string='', dim='sek'):
    
    '''
    Displays the time difference from a specified start time and the current 
    time in the console.
    
    *args:
        start_time: time.time() object of the built-in Python module time\n
        string: message displayed together with time difference\n
        dim: choose an output format as follows\n
              ms  - milli seconds\n
              mus - micro seconds\n
              MS  - min:sec (default)\n
              HMS - hr:min:sec
    '''

    t_now       = time.time()
    t_elapsed   = t_now-start_time
    
    if dim=='HMS':
        tt = time.strftime('%H:%M:%S', time.gmtime(t_elapsed))
        print(f'{string} - {tt} hrs')
        
    if dim=='MS':
        tt = time.strftime('%M:%S', time.gmtime(t_elapsed))
        print(f'{string} - {tt} min')
    
    if dim=='ms':
        t_ms        = round(t_elapsed*1000, 2)
        print(f'{string} - {t_ms} ms')
    
    if dim=='mus':
        t_mus        = round(t_elapsed*1000 * 1000, 2)
        print(f'{string} - {t_mus} \u03BCs')

###############################################################################
### double check if parameters were set right in the config dictionary ########

def troubleshooting(config):
    
    if config['source']['data']['type'] == 'atlas':
        pass
        #sys.exit('ATTENTION: Using synthetic obs data is not recommended.')

    if config['PSO_spec']['PSO_math']['n_mod_vec'] == 1:
        sys.exit('ATTENTION: The inversion using only one model vector is not fully implemented yet.')
        
    if config['PSO_spec']['PSO_math']['lp_norm'] == 'L1':
        sys.exit('ATTENTION: Using the LP-norm L1 is not fully implemented yet.')
            
    T1  = config['source']['signal']['T1']
    T2  = config['source']['signal']['T2']
    T2S = config['source']['signal']['T2S']
    SI  = config['source']['signal']['SI']
    JI  = config['source']['signal']['JI']
    
    # if T1:
    #     sys.exit('ATTENTION: No T1-measurements in current data set available.')
    
    if sum([T1, T2, T2S]) >= 2 and SI:
        print('ATTENTION: To many signals for a single inversion. Setting inversion to T2 signal.')
        if input('Continue? [y,n]: ') == 'n': sys.exit()
        else: 
            config['source']['signal']['T1'] = False
            config['source']['signal']['T2S'] = False
        
    if sum([T1, T2, T2S]) == 1 and JI:
        print('ATTENTION: Only one signal given for joint inversion. Setting inversion to T2+T2S signals.')
        if input('Continue? [y,n]: ') == 'n': sys.exit()
        else:
            config['source']['signal']['T1']  = False
            config['source']['signal']['T2']  = True
            config['source']['signal']['T2S'] = True

    if sum([T1, T2, T2S]) > 2 and JI:
        print('ATTENTION: More than 2 signals given for joint inversion'
              '(this case is not implemented yet).\n Setting inversion to T2+T2S signals.')
        if input('Continue? [y,n]: ') == 'n': sys.exit()
        else:
            config['source']['signal']['T1']  = False
            config['source']['signal']['T2']  = True
            config['source']['signal']['T2S'] = True
            
    if SI and JI:
        sys.exit('ATTENTION: Cannot perform both single and joint inversion at the same time.')

    if not SI and not JI:
        sys.exit('ATTENTION: Didn´t you forget something? Both inversion types are still set as false.')
        
    if JI and config['PSO_spec']['comp_mode']['iter_test']['use']:
        sys.exit('ATTENTION: Iteration test only possible for a single signal T1, T2, or T2S.')

    if config['PSO_spec']['PSO_math']['n_part'] % config['PSO_spec']['algo_math']['batch'] != 0:
        batch_size = config['PSO_spec']['algo_math']['batch']
        sys.exit(f'ATTENTION: Invalid number of particles. Please choose a multiple of the used batch size of {batch_size}!')
        
    return config
    
def open_parameters(path: str):
    
    ''' 
    Returns dictionary of parameters from a configuration file. Supported file formats are json, yaml.
    
    *args:
        path: absoulte path to the configuration file
    
    Returns:
        Dictionary with file content
    '''
    
    if path.endswith('.yaml') or path.endswith('.yml'):
        with open(path, 'r') as file:
            return yaml.safe_load(file)
            
    elif path.endswith('.json'):
        with open(path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError("Unsupported file format."
                         "Please provide a YAML (.yaml, .yml) or JSON (.json) file.")

def make_grid(Tmin, Tmax, npoints, **kwargs):
    """
    Create a grid from Tmin to Tmax with npoints grid points.

    kwargs:
        log: logarithmic grid (default)
        lin: linear grid
    """
    if kwargs.get('mode', None) == 'lin':  # creating linear grid
        return np.linspace(Tmin, Tmax, npoints)

    if kwargs.get('mode', None) == 'dumb':  # creating dumb grid :^)
        temp1 = np.linspace(2, 14, 25)
        temp2 = np.linspace(56, 104, 25)
        return np.append(temp1, temp2)

    if kwargs.get('mode', None) == 'fit_gauss':
        """
        Fit a grid to the gaussian distribution underlying the created signal.

        Experimental mode. Careful: here npoints is number of points per area
        areas are given by 6sigma (3sigma to ecah side) for peaks and then the rest.

        Needs a gaussian to work (needs mu and sigma values)

        don't use this if you have peaks that overlap in the 3sigma area...it will fk up
        """

        gauss = kwargs.get('gauss', None)
        assert gauss is not None, "no gaussian given. aborting."

        mu_values = gauss.mu_values
        sigma_values = gauss.sigma_values

        # apparently this is how you initialise an empty array -.-
        grid = np.array([]).reshape(0, 3)
        lenctr = 0
        maxes = []
        mins = []
        for i in range(len(mu_values)):
            tempmin = max(mu_values[i] - 3 * sigma_values[i], Tmin)
            tempmax = min(mu_values[i] + 3 * sigma_values[i], Tmax)
            lenctr += tempmax - tempmin
            maxes.append(tempmax)
            mins.append(tempmin)
            # np.linspace(tempmin, tempmax, npoints, endpoint=True))
            grid = np.append(grid, np.linspace(tempmin, tempmax, npoints))
        np.sort(grid)  # just in case ...
        np.sort(maxes)
        np.sort(mins)
        lminus = Tmax - Tmin - lenctr  # the length without a peak

        corr = 0
        if maxes[-1] == Tmax:
            corr += 1
        if mins[0] == Tmin:
            corr += 1
        npoints = npoints + 1 - corr  # getting right number of points
        dminus = lminus / npoints  # average distance of points in peakless area

        rest = mins[0] - Tmin  # three variables used for skipping peaks:
        temp = Tmin
        j = 0

        for i in range(npoints + 1 - corr):  # number of points depends on endpoints or nah
            if i == 0:
                if rest > 0:
                    temp = Tmin
            elif j == len(mins):
                temp += dminus
            elif rest <= dminus:
                temp = maxes[j] + dminus - rest
                rest = temp - mins[j]
                j += 1
            else:
                temp += dminus
                rest = mins[j] - temp

            grid = np.append(grid, temp)

        return np.sort(grid)

    else:  # creating logarithmic grid
        alpha = Tmin - 1
        beta = np.log(Tmax - Tmin + 1) / (npoints - 1)
        grid = np.zeros(npoints)
        for i in range(npoints):
            grid[i] = alpha + np.exp(beta * i)
        return grid

###############################################################################
# Caluclation of Gaussian bell curves
#
# a) Full integration over the whole model room

def compute_gauss_full(x, sigma, mean, scale):
    
    """
    *args:
        x: np.array (1D)
            evaluation position (system grid)
        sigma: np.array
            standard deviation / width (σ)
        mean: np.array
            center of the Gaussian (μ)
        scale: np.array
            scaling factor representing the integral of the Gaussian
            
    Returns:
        Gaussian bell curve with shape dependent on arguments 
    """

    return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

# b) Partial integration intervall based over a Region of Interest
#    
#    i.  Computation of index intervalls. 
#        Embedded in ParticleSwarmOptimizer._compute_intervall_gauss (PSOworkflow.py).
    
#    ii. Numba-accelerated core functions for PSO-based MRI signal simulation and
#        non-numba equivalents. For max speed use/try: @njit(fastmath=True, cache=True)

@njit
def _compute_gaussian_njit(x, mean, fac_sig_sqrt, fac_pre):    
    """Internal Gaussian basis kernel evaluated on the system grid."""    
    dx  = x - mean
    return fac_pre * np.exp(-np.square(dx) * fac_sig_sqrt)

def _compute_gaussian_py(x, mean, fac_sig_sqrt, fac_pre):    
    """Internal Gaussian basis kernel evaluated on the system grid."""    
    dx  = x - mean
    return fac_pre * np.exp(-np.square(dx) * fac_sig_sqrt) 

@njit
def _compute_matmul_njit(gauss, sys_mat):    
    """Internal matrix multiplication kernel."""    
    return gauss @ sys_mat

def _compute_matmul_py(gauss, sys_mat):    
    """Internal matrix multiplication kernel."""    
    return gauss @ sys_mat

@njit
def _compute_CT2S_njit(MW_f, FW_f, phi, mat1_array, mat2_array, mult_CT2S):
    """Internal kernel for complex CT2* signal computation."""                   
    exp_MW  = np.exp(mult_CT2S * MW_f)
    exp_FW  = np.exp(mult_CT2S * FW_f)
    exp_phi = np.exp(-1j * phi)
    return (mat1_array * exp_MW + mat2_array * exp_FW) * exp_phi

def _compute_CT2S_py(MW_f, FW_f, phi, mat1_array, mat2_array, mult_CT2S):
    """Internal kernel for complex CT2* signal computation."""                   
    exp_MW  = np.exp(mult_CT2S * MW_f)
    exp_FW  = np.exp(mult_CT2S * FW_f)
    exp_phi = np.exp(-1j * phi)
    return (mat1_array * exp_MW + mat2_array * exp_FW) * exp_phi