# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate as inting  # scipy standard numeric integration tool


def T1_decay(t, T1):
    """T1 decay formula."""
    return 1 - 2 * np.exp(-t / T1)


def T1_decay_abs(t, T1):
    """Absolute value of T1_decay above."""
    return np.abs(1 - 2 * np.exp(-t / T1))


def T2_decay(t, T2):
    """T2 decay formula."""
    return np.exp(-t / T2)


def system_matrix_from_kernel(timepoints, T_grid, kern):
    """
    Take values timepoints for time and T_grid for decay parameters and output a Matrix.

    Example:
    A[i][j]=exp(-t_values[i]/T_values[j])
    As described in Hannas Thesis.

    A needs to have this shape for the scipy.optimize.nnls to work with it.

    CAUTION: Make sure to not input integer times or the matrix will be incorrect
    """
    m = len(timepoints)  # initiating m and n for the sake of clarity
    n = len(T_grid)

    A = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            A[i][j] = kern(timepoints[i], T_grid[j])
    return A


def signal_from_distribution(A, t, Tmin, Tmax, kern):
    """Create signal for a given amplitude distribution A and positions (timepoints) t[i].

    Inputs:
        A: Amplitude distribution 
            Example: T1 distribution
        t: np.array
            timepoints
        Tmin: float
            Minimum T from which integration is performed            
        Tmax: float
            Maximum T until which integration is performed
        kern: kernel
            decay law which the signal follows.
    """
    n_t = len(t)
    signal = np.empty(n_t)

    def integrand(A, t):
        """
        Take a amplitude distribution M0(T1) and returns the kernel to be integrated.

        Example Integrand has the form:
        integrand(M0,t)(T1)=M0(T1)(1 - 2 * exp(-t/T1))

        T: decay constant
        t: time
        A: Amplitude distribution
        """
#        ED: ???
#        if t == 0:  # to prevent 0 division spaghetti from e^{-t/T} part of kernel
#            return lambda T: kern(0, 1) * A(T)
#        else:
    
        return lambda T: A(T) * kern(t, T)

    for i in range(n_t):
        signal[i] = inting.quad(integrand(A, t[i]), Tmin, Tmax)[0]
        


    return signal
