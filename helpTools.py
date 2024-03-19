"""
helper functions for T1, T2star and T2 MWF mapping of in vivo data
"""

# had to be installed: nibabel, natsort
# a) nibabel can open the typical MRT imaging format .nii
# --> NIfTI-Format (Neuroimaging Informatics Technology Initiative)
# b) natsort sorts a list for natural keys, such as integers
# --> MEMO: ich habe sowas schonmal für Uli geschrieben - raussuchen

# had to be installed: EPYG
# Extended Phase Graphs in Python (by Daniel Brenner, Bonn, 2016)
# --> https://github.com/JIMM-MRI/EpyG

import glob, os, natsort, sys
import numpy                       as     np
from   scipy                       import integrate as inting
from   scipy.ndimage.interpolation import shift
import nibabel                     as     nib
from   scipy.optimize              import nnls


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

def gauss(x, sigma, mean, scale):
    return scale / (np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x - mean) / (2 * sigma ** 2))    