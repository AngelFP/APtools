"""This module defines methods performing common operations and
equations for plasma-based accelerators"""

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
    return np.sqrt(ct.e**2 * plasma_dens*1e6 / (ct.m_e*ct.epsilon_0))

def laser_frequency(l_lambda):
    """Calculate the laser frequency from its wavelength

    Parameters:
    -----------
    l_lambda : float
        The laser wavelength in meters

    Returns:
    --------
    A float with laser frequency in units of 1/s
    """
    return 2*ct.pi*ct.c / l_lambda

def plasma_skin_depth(plasma_dens):
    """Calculate the plasma skin depth from its densiy

    Parameters:
    -----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns:
    --------
    A float with the plasma skin depth in meters
    """
    return ct.c / plasma_frequency(plasma_dens)

def plasma_focusing_gradient_blowout(n_p):
    """Calculate the plasma focusing gradient assuming blowout regime

    Parameters:
    -----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns:
    --------
    A float with the focusing gradient value in T/m
    """
    return ct.m_e*plasma_frequency(n_p)**2 / (2*ct.e*ct.c)

def self_guiding_threshold_a0_blowout(plasma_dens, l_lambda):
    """Get minimum a0 to fulfill self-guiding condition in the blowout regime.

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

def plasma_density_for_self_guiding_blowout(w_0, a_0, l_0=None):
    """Get plasma density fulfilling self-guiding condition in blowout regime.

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
        a_0_thres = self_guiding_threshold_a0_blowout(n_p, l_0)
        if a_0 < a_0_thres:
            print("Warning: laser a0 does not meet self-guiding conditions.")
            print("Value provided: {}, threshold value: {}".format(a_0,
                                                                   a_0_thres))
    return n_p

def laser_w0_for_self_guiding_blowout(n_p, a_0, l_0=None):
    """Get plasma density fulfilling self-guiding condition in blowout regime.

    For more inforation see W. Lu - 2007 - Generating multi-GeVelectron bunches
    using single stage laser wakeﬁeld acceleration in a 3D nonlinear regime
    (https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.10.061301)

    Parameters:
    -----------
    n_p : float
        The plasma density in units of cm-3

    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength. Only necessary to check that the self-guiding
        threshold is met.

    Returns:
    --------
    A float with the value of w_0 in meters
    """
    k_p = plasma_frequency(n_p) / ct.c
    w_0 = 2 * np.sqrt(a_0)/k_p
    if l_0 is not None:
        a_0_thres = self_guiding_threshold_a0_blowout(n_p, l_0)
        if a_0 < a_0_thres:
            print("Warning: laser a0 does not meet self-guiding conditions.")
            print("Value provided: {}, threshold value: {}".format(a_0,
                                                                   a_0_thres))
    return w_0

def matched_beam_size(beam_ene, beam_em, n_p=None, k_x=None):
    """Get matched beam size for the plasma focusing fields.

    The focusing gradient, k_x, can be provided or calculated from the plasma
    density, n_p.

    Parameters:
    -----------
    beam_ene : float
        Unitless electron beam mean energy (beta*gamma)

    beam_em : float
        The electron beam normalized emittance in m*rad

    n_p : float
        The plasma density in units of cm-3

    k_x : float
        The plasma transverse focusing gradient in T/m

    Returns:
    --------
    A float with the value of beam size in meters
    """
    if k_x == None:
        if n_p == None:
            raise ValueError("No values for the plasma density and focusing"
                             " gradient have been provided.")
        else:
            k_x = plasma_focusing_gradient_blowout(n_p)
    # betatron frequency
    w_x = np.sqrt(ct.c*ct.e/ct.m_e * k_x/beam_ene)
    # matched beta function
    b_x = ct.c/w_x
    # matched beam size
    s_x = np.sqrt(b_x*beam_em/beam_ene)
    return s_x
