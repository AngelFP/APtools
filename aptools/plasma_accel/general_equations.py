"""This module defines methods performing common operations and
equations for plasma-based accelerators"""

import scipy.constants as ct
import numpy as np


def plasma_frequency(plasma_dens):
    """Calculate the plasma frequency from its densiy

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with plasma frequency in units of 1/s
    """
    return np.sqrt(ct.e**2 * plasma_dens*1e6 / (ct.m_e*ct.epsilon_0))


def plasma_skin_depth(plasma_dens):
    """Calculate the plasma skin depth from its densiy

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with the plasma skin depth in meters
    """
    return ct.c / plasma_frequency(plasma_dens)


def plasma_wavelength(plasma_dens):
    """Calculate the plasma wavelength from its densiy

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with the plasma wavelength in meters
    """
    return 2*ct.pi*ct.c / plasma_frequency(plasma_dens)


def plasma_cold_non_relativisct_wave_breaking_field(plasma_dens):
    """Calculate the cold, non relativisitic wave breaking field from the
    plasma density.

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with the field value in V/m
    """
    return ct.m_e*ct.c/ct.e * plasma_frequency(plasma_dens)


def plasma_focusing_gradient_blowout(n_p):
    """Calculate the plasma focusing gradient assuming blowout regime

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with the focusing gradient value in T/m
    """
    return ct.m_e*plasma_frequency(n_p)**2 / (2*ct.e*ct.c)


def plasma_focusing_gradient_linear(n_p, dist_from_driver, a_0, w_0):
    """Calculate the plasma focusing gradient assuming linear regime.

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    dist_from_driver : string
        Distance from the driver center in units of m.

    a_0 : float
        Peak normalized vector potential of the laser.

    w_0 : float
        Spot size (w_0) of the laser pulse in units of m.

    Returns
    -------
    A float with the focusing gradient value in T/m.

    """
    w_p = plasma_frequency(n_p)
    k_p = w_p/ct.c
    E_0 = ct.m_e*ct.c*w_p/ct.e
    K = (8*np.pi/np.e)**(1/4)*a_0/(k_p*w_0)
    return -E_0*K**2*k_p*np.sin(k_p*dist_from_driver)/ct.c


def laser_frequency(l_lambda):
    """Calculate the laser frequency from its wavelength.

    Parameters
    ----------
    l_lambda : float
        The laser wavelength in meters

    Returns
    -------
    A float with laser frequency in units of 1/s
    """
    return 2*ct.pi*ct.c / l_lambda


def laser_rayleigh_length(w_0, l_lambda):
    """Calculate the Rayleigh length of the laser assuming a Gaussian profile.

    Parameters
    ----------
    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    l_lambda : float
        The laser wavelength in meters

    Returns
    -------
    A float with Rayleigh length in m
    """
    return ct.pi * w_0**2 / l_lambda


def laser_radius_at_z_pos(w_0, l_lambda, z):
    """Calculate the laser radius (W) at a distance z from its focal position.

    Parameters
    ----------
    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    l_lambda : float
        The laser wavelength in meters.

    z : float or array
        Distance from the focal position (in meters) at which to calculate the
        laser radius.

    Returns
    -------
    A float or array with laser radius (W) in meters
    """
    z_r = laser_rayleigh_length(w_0, l_lambda)
    return w_0 * np.sqrt(1 + (z/z_r)**2)


def self_guiding_threshold_a0_blowout(plasma_dens, l_lambda):
    """Get minimum a0 to fulfill self-guiding condition in the blowout regime.

    For more details see W. Lu - 2007 - Designing LWFA in the blowout regime
    (https://ieeexplore.ieee.org/iel5/4439904/4439905/04440664.pdf).

    Parameters
    ----------
    plasma_dens : float
        The plasma density in units of cm-3

    l_lambda : float
        The laser wavelength in meters

    Returns
    -------
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

    Parameters
    ----------
    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength in meters. Only necessary to check that the
        self-guiding threshold is met.

    Returns
    -------
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


def laser_energy(a_0, l_0, lon_fwhm, w_0):
    """Calculate laser pulse energy assuming Gaussian profile.

    Parameters
    ----------
    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength in meters.

    lon_fwhm : float
        Longitudinal FWHM of the intensity in seconds.

    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    Returns
    -------
    A float with the value of the energy in Joules
    """
    i_peak = 2*np.pi**2*ct.epsilon_0*ct.m_e**2*ct.c**5 * a_0**2 / (ct.e*l_0)**2
    s_x = w_0 / 2
    s_y = s_x
    s_z = lon_fwhm / (2*np.sqrt(2*np.log(2)))
    l_ene = (2*np.pi)**(3/2) * s_x * s_y * s_z * i_peak
    return l_ene


def laser_peak_intensity(a_0, l_0, z=None, w_0=None):
    """Calculate laser pulse peak intensity assuming Gaussian profile.

    Parameters
    ----------
    a_0 : float
        The laser a_0.

    l_0 : float
        The laser wavelength in meters.

    z : float or array
        Distance to the focal position of the laser pulse.

    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)). Only needed if
        z is not None.

    Returns
    -------
    A float with the value of the peak power in W/m^2.
    """
    if z is not None:
        z_r = laser_rayleigh_length(w_0, l_0)
        a_peak = a_0 / np.sqrt(1 + (z/z_r)**2)
    else:
        a_peak = a_0
    k = 2*np.pi**2*ct.epsilon_0*ct.m_e**2*ct.c**5
    i_peak = k * a_peak**2 / (ct.e*l_0)**2
    return i_peak


def laser_peak_power(a_0, l_0, w_0):
    """Calculate laser pulse peak power assuming Gaussian profile.

    Parameters
    ----------
    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength in meters.

    w_0 : float
        The laser beam waist in meters, i. e., 1/e in field or 1/e^2 in
        intesity. Calculate from FWHM as FWHM/sqrt(2*log(2)).

    Returns
    -------
    A float with the value of the oeak power in Watts
    """
    i_peak = laser_peak_intensity(a_0, l_0)
    p_peak = np.pi * w_0**2 * i_peak / 2
    return p_peak


def laser_w0_for_self_guiding_blowout(n_p, a_0, l_0=None):
    """Get laser spot size fulfilling self-guiding condition in blowout regime.

    For more inforation see W. Lu - 2007 - Generating multi-GeV electron
    bunches using single stage laser wakeﬁeld acceleration in a 3D nonlinear
    regime (https://journals.aps.org/prab/pdf/10.1103/PhysRevSTAB.10.061301)

    Parameters
    ----------
    n_p : float
        The plasma density in units of cm-3

    a_0 : float
        The laser a_0

    l_0 : float
        The laser wavelength in meters. Only necessary to check that the
        self-guiding threshold is met.

    Returns
    -------
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


def matched_laser_pulse_duration_blowout(n_p, a_0):
    """Get maximum matched laser pulse duration in the blowout regime.

    For more details see W. Lu - 2007 - Designing LWFA in the blowout regime
    (https://ieeexplore.ieee.org/iel5/4439904/4439905/04440664.pdf).

    Parameters
    ----------
    n_p : float
        The plasma density in units of cm-3

    a_0 : float
        The laser a_0

    Returns
    -------
    A float with the value of t_FWHM in seconds
    """
    k_p = plasma_frequency(n_p) / ct.c
    t_FWHM = 2/ct.c * np.sqrt(a_0)/k_p
    return t_FWHM


def matched_beam_size(beam_ene, beam_em, n_p=None, k_x=None):
    """Get matched beam size for the plasma focusing fields.

    The focusing gradient, k_x, can be provided or calculated from the plasma
    density, n_p.

    Parameters
    ----------
    beam_ene : float
        Unitless electron beam mean energy (beta*gamma)

    beam_em : float
        The electron beam normalized emittance in m*rad

    n_p : float
        The plasma density in units of cm-3

    k_x : float
        The plasma transverse focusing gradient in T/m

    Returns
    -------
    A float with the value of beam size in meters
    """
    # matched beta function
    b_x = matched_plasma_beta_function(beam_ene, n_p, k_x)
    # matched beam size
    s_x = np.sqrt(b_x*beam_em/beam_ene)
    return s_x


def matched_plasma_beta_function(beam_ene, n_p=None, k_x=None,
                                 regime='Blowout', dist_from_driver=None,
                                 a_0=None, w_0=None):
    """Get beta function from the plasma focusing fields.

    The focusing gradient, k_x, can be provided or calculated from the plasma
    density, n_p.

    Parameters
    ----------
    beam_ene : float
        Unitless electron beam mean energy (beta*gamma)

    n_p : float
        The plasma density in units of cm-3

    k_x : float
        The plasma transverse focusing gradient in T/m

    regime : string
        Specify the accelation regime ('Linear' or 'Blowout') for which to
        calculate the focusing fields. Only used if k_x is not provided.

    dist_from_driver : string
        Distance from the driver center in units of m. Only needed for Linear
        regime.

    a_0 : float
        Peak normalized vector potential of the laser. Only needed for Linear
        regime.

    w_0 : float
        Spot size (w_0) of the laser pulse in units of m. Only needed for
        Linear regime.

    Returns
    -------
    A float with the value of the beta function in meters

    """
    if k_x is None:
        if n_p is None:
            raise ValueError("No values for the plasma density and focusing"
                             " gradient have been provided.")
        else:
            if regime == 'Blowout':
                k_x = plasma_focusing_gradient_blowout(n_p)
            elif regime == 'Linear':
                k_x = plasma_focusing_gradient_linear(n_p, dist_from_driver,
                                                      a_0, w_0)
            else:
                raise ValueError("Unrecognized acceleration regime")
    # betatron frequency
    w_x = np.sqrt(ct.c*ct.e/ct.m_e * k_x/beam_ene)
    # beta function
    b_x = ct.c/w_x
    return b_x


def maximum_wakefield_plasma_lens(q_tot, s_z, s_r, n_p):
    """Calculates the maximum focusing gradient induced by beam wakefields in
    an active plasma lens using linear theory.

    Formula obtained from https://arxiv.org/pdf/1802.02750.pdf

    Parameters
    ----------
    q_tot : float
        Total beam charge in C

    s_z : float
        RMS beam length in m

    s_r : float
        RMS beam size in m

    n_p : float
        The plasma density in units of cm-3

    Returns
    -------
    A float with the value of the focusing gradient in T/m
    """
    k_p = 1 / plasma_skin_depth(n_p)
    a = 1 + k_p**2 * s_r**2 / 2
    b = 1 + np.sqrt(8*ct.pi) * k_p**2 * s_z**2
    g_max = q_tot * ct.mu_0 * ct.c * k_p**2 * s_z / (2*ct.pi * s_r**2 * a * b)
    return g_max
