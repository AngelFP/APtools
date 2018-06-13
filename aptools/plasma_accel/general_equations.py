import scipy.constants as ct
import numpy as np

def plasma_frequency(plasma_dens):
    """Calculate the plasma frequency from its densiy
    
    Parameters:
    -----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns:
    --------
    A float with plasma frequency in units of 1/s
    """
    return ct.e * np.sqrt(plasma_dens*1e6 / (ct.m_e*ct.epsilon_0))

def laser_frequency(l_lambda):
    """Calculate the laser frequency from its wavelength
    
    Parameters:
    l_lambda : float
        The laser wavelength in meters

    Returns:
    --------
    A float with laser frequency in units of 1/s
    """
    return 2*ct.pi*ct.c / l_lambda

def self_guiding_threshold_a0(plasma_dens, l_lambda):
    """Get the minimum a0 to fulfill the self-guiding condition.

    For more details see W. Lu - 2007 - Designing LWFA in the blowout regime
    (https://ieeexplore.ieee.org/iel5/4439904/4439905/04440664.pdf).

    Parameters:
    -----------
    plasma_dens : float
        The plasma density in units of cm-3

    l_lambda : float
        The laser wavelength in meters

    Returns:
    --------
    A float with the value of the threshold a0
    """
    w_p = plasma_frequency(plasma_dens)
    w_0 = laser_frequency(l_lambda)
    a_0 = (w_0/w_p)**(2/5)
    return a_0

def plasma_density_for_self_guiding(w_0, a_0, l_0=None):
    """Get the plasma density to fulfill the self-guiding condition.

    For more inforation see W. Lu - 2007 - Generating multi-GeVelectron bunches
    using single stage laser wakeﬁeld acceleration in a 3D nonlinear regime
    (https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.10.061301)


    Parameters:
    -----------
    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength. Only necessary to check that the self-guiding
        threshold is met.

    Returns:
    --------
    A float with the value of the plasma density in units of cm-3
    """
    k_p = 2 * np.sqrt(a_0)/w_0
    n_p = k_p**2 * ct.m_e*ct.epsilon_0*ct.c**2/ct.e**2 * 1e-6
    if l_0 is not None:
        a_0_thres = self_guiding_threshold_a0(n_p, l_0)
        if a_0 < a_0_thres:
            print("Warning, laser a0 does not meet self-guiding conditions.")
            print("Value provided: {}, threshold value: {}".format(a_0,
                                                                   a_0_thres))
    return n_p
