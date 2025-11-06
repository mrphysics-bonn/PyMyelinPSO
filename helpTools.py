"""
helper functions for T1, T2star and T2 MWF mapping of in vivo data
"""

# had to be installed: nibabel, natsort
# a) nibabel can open the typical MRT imaging format .nii
# --> NIfTI-Format (Neuroimaging Informatics Technology Initiative)
# b) natsort sorts a list for natural keys, such as integers
# --> MEMO: ich habe sowas schonmal für Uli geschrieraussuchen
            
# had to be installed: EPYG
# Extended Phase Graphs in Python (by Daniel Brenner, Bonn, 2016)
# --> https://github.com/JIMM-MRI/EPYG

import os, natsort, sys, yaml, json, time, matplotlib
import numpy                       as     np
import numba                       as     nb
import nibabel                     as     nib
import numexpr                     as     ne

from   functools                   import partial
from   scipy                       import integrate as inting
from   scipy.ndimage.interpolation import shift
from   scipy.optimize              import nnls

from   mwf_t1t2t2s                 import mwf_analysis
from   PSOparameters               import Parameters   as PM

import matplotlib.pyplot           as     plt
from   matplotlib.widgets          import RectangleSelector

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

    T_now       = time.time()
    T_elapsed   = T_now-start_time
    
    if dim=='HMS':
        TT = time.strftime('%H:%M:%S', time.gmtime(T_elapsed))
        print(f'{string} - {TT} hrs')
        
    if dim=='MS':
        TT = time.strftime('%M:%S', time.gmtime(T_elapsed))
        print(f'{string} - {TT} min')
    
    if dim=='ms':
        t_ms        = round(T_elapsed*1000, 2)
        print(f'{string} - {t_ms} ms')
    
    if dim=='mus':
        t_mus        = round(T_elapsed*1000 * 1000, 2)
        print(f'{string} - {t_mus} \u03BCs')

# MWF_analysis class initialisation to gain access to maiden methods/objects
# !!! KW_T2SP file not being used, but if not given as argument, an error occurs: 
            # File ~\Project JIMM\Python\PyJIMM\mwf_t1t2t2s.py:331 in prep_data
            # if (self.slice['T2S'].any()) and (self.slice['CT2S'].any()):
# update - phase information use is being implemented now


class create_system_values(PM):
    
    def __init__(self, config, noSlice):
        
        super().__init__()
            
        rootMWF = mwf_analysis(data_dir = config['source']['file']['dir'],
                               KW_B1    = config['source']['file']['B1'],
                               KW_T1    = config['source']['file']['T1'],
                               KW_T2    = config['source']['file']['T2'],
                               KW_T2S   = config['source']['file']['T2S'],
                               KW_T2SP  = config['source']['file']['T2SP'],
                               T2S_TE   = config['source']['file']['T2S_TE'])
    
        # rootMWF.prep_data(axis        = 'Z', 
        #                   slice_num   = noSlice, 
        #                   signal_type = 'T1',
        #                   filter      = 1.0, 
        #                   thresh      = 4.5, 
        #                   verbose     = False)
    
        rootMWF.prep_t1_model(tr      = self.Inversion.T1_TR,
                              alpha   = self.Inversion.T1_alpha,
                              td      = self.Inversion.T1_TD,
                              ie      = self.Inversion.T1_IE,
                              T1min   = self.Inversion.T1min,
                              T1max   = self.Inversion.T1max,
                              nT1     = self.Inversion.modSpace,
                              verbose = False)
        
        rootMWF.prep_data(axis        = 'Z',
                          slice_num   = noSlice,
                          signal_type = 'T2',
                          filter      = 1.0,
                          thresh      = 4.5,
                          verbose     = False)
    
        rootMWF.prep_t2_model(te      = self.Inversion.T2_TE,
                              tr      = self.Inversion.T2_TR,
                              etl     = self.Inversion.datSpaceT2,
                              alpha   = self.Inversion.T2_alpha,
                              beta    = self.Inversion.T2_beta,
                              T2min   = self.Inversion.T2min,
                              T2max   = self.Inversion.T2max,
                              nT2     = self.Inversion.modSpace,
                              T1      = self.Inversion.T2_T1,
                              verbose = False)
        
        rootMWF.prep_data(axis        = 'Z', 
                          slice_num   = noSlice, 
                          signal_type = 'T2S',
                          filter      = 1.0, 
                          thresh      = 4.5, 
                          verbose     = False)
    
        rootMWF.prep_t2s_model(T2Smin  = self.Inversion.T2Smin,
                               T2Smax  = self.Inversion.T2Smax,
                               nT2S    = self.Inversion.modSpace,
                               verbose = False)

        self.rootMWF = rootMWF


def troubleshooting(config):
    
    if config['source']['data']['obs_data'] == 'synt':
        sys.exit('ATTENTION: Using synthetic obs data is not recommended.')

    if config['PSO_spec']['mod_vec'] == 1:
        sys.exit('ATTENTION: The inversion using only one model vector is not fully implemented yet.')
        
    if config['PSO_spec']['lp_norm'] == 'L1':
        sys.exit('ATTENTION: Using the LP norm L1 is not fully implemented yet.')
            
    T1  = config['source']['signal']['T1']
    T2  = config['source']['signal']['T2']
    T2S = config['source']['signal']['T2S']
    SI  = config['source']['signal']['SI']
    JI  = config['source']['signal']['JI']
    
    if T1 == True:
        sys.exit('ATTENTION: No T1-measurements in current data set available.')
    
    if sum([T1, T2, T2S]) >= 2 and SI == True:
        print('ATTENTION: To many parameters for a single inversion. Setting inversion to T2 signal.')
        if input('Continue? [y,n]: ') == 'n': sys.exit()
        else: 
            config['source']['signal']['T1'] = False
            config['source']['signal']['T2S'] = False
        
    if sum([T1, T2, T2S]) == 1 and JI == True:
        print('ATTENTION: Only one signal given for joint inversion. Setting inversion to T2+T2S signals.')
        if input('Continue? [y,n]: ') == 'n': sys.exit()
        else:
            config['source']['signal']['T1']  = False
            config['source']['signal']['T2']  = True
            config['source']['signal']['T2S'] = True

    if SI == True and JI == True:
        sys.exit('ATTENTION: Cannot perform both single and joint inversion at the same time.')

    if SI == False and JI == False:
        sys.exit('ATTENTION: Didn´t you forget something? Both inversion types are still set as false.')
        
    # if JI == True and noIterTest == True:
    #     sys.exit('ATTENTION: Iteration test only possible for a single signal T1, T2, or T2S.')

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

def get_synData(filelist: list, noSlice: int, indShift=0, meta=False):

    """ Stores synthetic MWF data into a numpy array """
    
    data             = filelist[0].get_fdata()
    data_img         = data[:, :, noSlice]
    index_i, index_j = np.unravel_index(np.argmax(data_img), data_img.shape)
    data_px          = data[index_i, index_j + indShift, noSlice]

    if meta == True:
        print(f'Array size: {data.shape} - (MxN, NoImg)\n')
    
    return data, data_px, index_i, index_j + indShift

def get_measData(filename: list):
    
    """ Stores measured decay data into a numpy array """
    
    if filename == 'NO_FILE':
        return

    return filename.get_fdata()

def get_maskData(filename: str):
    
    """ Stores mask data into a numpy array """
    
    
    
    
def load_data(path: str, filename: str, NIFTI_KEY=None):

    """ Loads MRI data into a filelist
    
    Args:
        path: working directory
        filename:  file name
        NIFTI_KEY: keywords to search multiple images in WD
    
    RETURNS
        nimg: list of GRE nifti images (grey)
    """
    
    if filename == 'NO_FILE':
        return ['NO_FILE']

    # filelist  = glob.glob(path+f'\\*{ext}', recursive=True)
    
    filelist  = [os.path.join(path, file) for file in os.listdir(path) if os.path.basename(file) == filename]

    if NIFTI_KEY:
        nifti = [s for s in filelist if NIFTI_KEY in s]
        nifti = natsort.natsorted(nifti)
    
        if len(nifti)==0:
            raise Exception(f'no images found using key: {NIFTI_KEY}')    
    else:
        nifti = filelist
        
    nimg = [None]*len(nifti)
    
    for i, fname in enumerate(nifti, start=0):
        nimg[i] = nib.load(os.path.join(path,fname))

    return nimg

def load_PSO_data(path: str, ext: str, key=None):
    
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.normpath(os.path.join(root, filename)))
    return files

def nnls_reg_simple(A, b, beta , p=0.0):
    """
    solving min_x ||A*x - b||^2 + beta^2||x-p||^2 by letting C=[A, beta*I] and d=[b, beta*p], 
    then solving min_x ||C*x - d||^2 with scipy.optimize.nnls. beta is the regularization 
    parameter and p is a constant prior for the solution, which must be either scalar or 
    of same dimension as solution x.
    """
    if beta == 0.0: return nnls(A, b)
    I = np.identity(np.size(A, axis=1))
    C = np.concatenate((A, beta*I), axis=0)
    if np.isscalar(p): p = p*np.ones(np.size(A, axis=1))
    x = np.concatenate((b, beta*p), axis=0)
    return nnls(C, x)
    
def mwf_estimate(signal, T2_grid, A, beta =0.0, T2_thresh = 20):
    
    # run regularized NNLS
    weights, chi2 = nnls_reg_simple(A, signal, beta)
        
    fast_comp = T2_grid < T2_thresh
    slow_comp = T2_grid > T2_thresh
    weights_area = np.sum(weights[fast_comp] ), np.sum(weights[slow_comp] )
    
    return (weights_area[0] / np.sum(weights_area) , weights)


class gauss_sum:
    r"""
    Holds array of Gaussian normal distributions which can be summed.

    a class that saves an array of medium values mu, standard deviations sigma and weights w and
    represents the function
    $a(T)=\sum_{i=0}^{n} w_i/\sqrt{2\pi\sigma^2}e^{-(\mu_i-T)^2/(2\sigma^2)}$
    """

    def __init__(self, MV, SD, W):
        """Initialise gauss sum."""
        assert len(MV) == len(W) == len( SD), 'parameter numbers not matching, cannot create gaussian sum.'
        self.sigma_values = SD  # array containing standard deviations
        self.mu_values = MV  # array containing medium values
        self.weights = W  # containing the weights of the individual distributions
        self.n = len(MV)  # number of peaks
        self.Tmin, self.Tmax = self.integration_limits()

    def __call__(self, T):
        """Evaluate the gaussian sum contained in this class at position (time) T."""
        sumvalue = 0
        for i in range(0, self.n):
            sumvalue += self.weights[i] / np.sqrt(2 * np.pi * self.sigma_values[i]**2) * \
                np.exp(-(T - self.mu_values[i])**2 / (2 * self.sigma_values[i]**2))

        # norming sum so that the integral of the function is (pretty much) 1 at t=0
        return sumvalue / np.sum(self.weights)

    def integration_limits(self):
        """
        Make an educated guess for integration limits for the given function.

        Gets the integration limits for a gaussian sum with standard deviations sigma_values and averages mu_values
        so that NNLS will look round and (almost) all of the mass of the function is contained in the integration
        limits.
        """
        Tmin_helper = self.mu_values - 10 * \
            self.sigma_values  # Setting integration boundaries so that all gauss peaks are almost fully contained,
        # 10 sigma away from the peak, not much should be left of a normal distribution.
        Tmax_helper = self.mu_values + 10 * self.sigma_values
        # calculating starting decay time T for integration.
        Tmin = max(1e-16, np.min(Tmin_helper, axis=0))
        # CAVE: The 1 for the minimum time might need to be changed (ED: Changed 1e-16), 
        # it was chosen to prevent 0 division as it is probably
        # physically plausible to assume $T_{min}\geq 1$ for all components of a human brain within the
        # abovementioned 10*sigma area
        # ending decay time T for integration
        Tmax = max(1.5 * np.max((self.mu_values), axis=0), np.max(Tmax_helper, axis=0))
        return Tmin, Tmax

    def calculate_convergent_weights(self, T2_grid, **kwargs):
        """
        Calculate convergent weights for comparison with reconstruction via NNLS.

        Careful! The T2 grid should be chosen correctly! Default is a logarithmic grid.
        """
        # i think these 2 lines are wrong, leaving 'em for historical reasons xD
        Tmin = self.Tmin
        Tmax = self.Tmax

        convergent_weights = np.zeros(len(T2_grid))
        # the first and last weight need to be treated differently
        convergent_weights[0] = inting.quad(self.__call__, Tmin, np.mean([Tmin, T2_grid[1]]))[0]
        convergent_weights[-1] = inting.quad(self.__call__,
                                             np.mean([T2_grid[-2], T2_grid[-1]]), Tmax)[0]
        for i in range(1, len(convergent_weights) - 1):
            # integrating from between the points hugely improves approximation of
            lower = np.mean([T2_grid[i - 1], T2_grid[i]])
            # decay generated by the convergent weights w.r.t. decay by gaussian
            upper = np.mean([T2_grid[i], T2_grid[i + 1]])
            convergent_weights[i] = inting.quad(self.__call__, lower, upper)[0]

        return convergent_weights

    def fast_convergent_weights(self, T2_grid, **kwargs):
        """Faster (>200x), but maybe slightly (<2/1000) inaccurate version of the above function.

        add kwarg 'lin' for linear grids to speed up.
        """
        convergent_weights = np.empty(len(T2_grid))

        convergent_weights = self.__call__(T2_grid)

        # fixing first and last entry
        convergent_weights[0] /= 2
        convergent_weights[-1] /= 2

        mode = kwargs.get('mode', None)

        # normalising
        # can be done faster for linear grid
        if mode == 'lin':
            convergent_weights /= np.sum(convergent_weights)
        else:
            shifted = shift(T2_grid, -1)
            normaliser = (shifted - T2_grid)
            normaliser[-1] = normaliser[-2]
            convergent_weights = convergent_weights * normaliser

        return convergent_weights


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


def add_rician_noise(signal, SNR, **kwargs):
    """
    Add rician noise to a real-valued signal with amplitude signal and given SNR.
    If the input signal is an array noise is scaled to first element of siganl array

    Note that SNR must be > 0, else 0 division.
    If mu is 0, the returned noise value will be 0 as well.

    kwargs:
    skip: skips noise addition

    returns:
    noisy signal
    """
    skip = kwargs.get('skip', None)

    if skip:
        return signal

    noisy_realpart = signal + np.random.normal(loc=0, scale=np.abs(signal[0]) / SNR, size=np.shape(signal))
    noisy_imagpart = np.random.normal(loc=0, scale=np.abs(signal[0]) / SNR, size=np.shape(signal))
    

    return np.sqrt(noisy_realpart**2 + noisy_imagpart**2)*np.sign(signal)


def signal_from_distribution(M0, t, T1_min, T1_max, kern):
    """Create signal for a given amplitude distribution M0 and timepoints t[i].

    Inputs:
        M0: kernel
            T1 distribution
        t: np.array
            timepoints
        T1_max: float
            Maximum T1 until which integration is performed
        kern: kernel
            decay law which the signal follows.
    """
    n_t = len(t)
    signal = np.empty(n_t)

    def integrand(M0, t):
        """
        Take a kernel M0(T1) and returns the kernel to be integrated.

        Inegrand has the form:
        integrand(M0,t)(T1)=M0(T1)(1 - 2 * exp(-t/T1))

        T1: decay constant
        t: time
        M0: Amplitude distribution
        """
        if t == 0:  # to prevent 0 division spaghetti from e^{-t/T1} part of kernel
            return lambda T1: kern(0, 1) * M0(T1)
        else:
            return lambda T1: M0(T1) * kern(t, T1)

    # for i in range(n_t):
    #     signal[i] = inting.quad(integrand(M0, t[i]), T1_min, T1_max)[0]
            
    signal = np.asarray([inting.quad(integrand(M0, t[i]), T1_min, T1_max)[0]
                            for i in range(n_t)])

    return signal


def T1_decay(t, T1):
    """T1 decay formula."""
    return 1 - 2 * np.exp(-t / T1)


def T1_decay_abs(t, T1):
    """Absolute value of T1_decay above."""
    return np.abs(1 - 2 * np.exp(-t / T1))


def T2_decay(t, T2):
    """T2 decay formula."""
    return np.exp(-t / T2)

# def gauss_part(x_sub, sigma, mean, scale, val=[None, None, None, None]):
#     '''
#     Creates the very same object as gauss_full(), but uses only maximum peak-intervals.
    
#     args:
#         x_sub - cutted system grid
#         val   - [left, right, model room size, factor]
#     '''
    
#     if val[1]-val[0]<val[2]:
#         exp_arg                 = -np.square(x_sub - mean) / (2 * sigma**2)
    
#         array                   = np.zeros((sigma.shape[0], val[2]))
#         array[:, val[0]:val[1]] = scale / (val[3]*sigma) * np.exp(exp_arg)

#         return array

#     else:
#        return scale / (val[3]*sigma) * np.exp(-np.square(x_sub - mean) / (2 * sigma ** 2)) 

def gauss_part(x, sigma, mean, scale, width=5):
    '''
    Creates the very same object as gauss_full(), but uses only maximum peak-intervals.
    
    args:
        x - system grid - and array which ...
    '''
    #######################################################################################
    ## approximating bell-curve width using sigma-width is best/fastest solution for now
    
    left  = np.searchsorted(x[0], np.min(mean) - width * np.max(sigma), side='left')
    right = np.searchsorted(x[0], np.max(mean) + width * np.max(sigma), side='right')
    x_sub = x[0, left:right]

    exp_arg = -np.square(x_sub[None,:] - mean) / (2 * sigma**2)
    
    array                = np.zeros((sigma.shape[0], x.shape[1]))
    array[:, left:right] = scale / (np.sqrt(2 * np.pi) * sigma) * np.exp(exp_arg)

    return array

# !!!! it seems numba becomes efficient with very big arrays with shape 10**6
# @nb.njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline="always")
def gauss_full(x, sigma, mean, scale):  
    return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

@nb.njit(fastmath=True, cache=True, boundscheck=False, nogil=True, inline="always")
def fast_matmul(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a @ b # or np.dot
    