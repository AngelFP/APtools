"""This module defines methods calculating beam properties"""

import scipy.constants as ct
import numpy as np

from aptools.helper_functions import (weighted_std, create_beam_slices,
                                      remove_correlation,
                                      calculate_slice_average)


def twiss_parameters(x, px, pz, py=None, w=None, emitt='tr',
                     disp_corrected=False, corr_order=1):
    """Calculate the alpha and beta functions of the beam in a certain
    transverse plane

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma). Necessary if
        disp_corrected=True or emitt='ph'.

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).

    w : array or single value
        Statistical weight of the particles.

    emitt : str
        Determines which emittance to use to calculate the Twiss parameters.
        Possible values are 'tr' for trace-space emittance and 'ph' for
        phase-space emittance

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    Returns
    -------
    A tuple with the value of the alpha, beta [m] and gamma [m^-1] functions
    """
    if emitt == 'ph':
        em_x = normalized_transverse_rms_emittance(x, px, py, pz, w,
                                                   disp_corrected, corr_order)
        gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
        gamma_avg = np.average(gamma, weights=w)
        x_avg = np.average(x, weights=w)
        px_avg = np.average(px, weights=w)
        # center x and x
        x = x - x_avg
        px = px - px_avg
        if disp_corrected:
            # remove x-gamma correlation
            dgamma = (gamma - gamma_avg)/gamma_avg
            x = remove_correlation(dgamma, x, w, corr_order)
        b_x = np.average(x**2, weights=w)*gamma_avg/em_x
        a_x = -np.average(x*px, weights=w)/em_x
    elif emitt == 'tr':
        em_x = transverse_trace_space_rms_emittance(x, px, py, pz, w,
                                                    disp_corrected, corr_order)
        xp = px/pz
        # center x and xp
        x_avg = np.average(x, weights=w)
        xp_avg = np.average(xp, weights=w)
        x = x - x_avg
        xp = xp - xp_avg
        if disp_corrected:
            # remove x-gamma correlation
            gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
            gamma_avg = np.average(gamma, weights=w)
            dgamma = (gamma - gamma_avg)/gamma_avg
            x = remove_correlation(dgamma, x, w, corr_order)
            # remove xp-gamma correlation
            xp = remove_correlation(dgamma, xp, w, corr_order)
        b_x = np.average(x**2, weights=w)/em_x
        a_x = -np.average(x*xp, weights=w)/em_x
    g_x = (1 + a_x**2)/b_x
    return (a_x, b_x, g_x)


def dispersion(x, px, py, pz, gamma_ref=None, w=None):
    """Calculate the first-order dispersion from the beam distribution

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma).

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).

    gamma_ref : float
        Reference energy for the dispersive element. If 'None' this will be the
        beam average energy.

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the value of the dispersion in m.
    """
    gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    if gamma_ref is None:
        gamma_ref = np.average(gamma, weights=w)
    dgamma = (gamma - gamma_ref)/gamma_ref
    fit_coefs = np.polyfit(dgamma, x, 1, w=w)
    disp = fit_coefs[0]
    return disp


def rms_length(z, w=None):
    """Calculate the RMS bunch length of the provided particle
    distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the RMS length value in meters.
    """
    s_z = weighted_std(z, weights=w)
    return s_z


def rms_size(x, w=None):
    """Calculate the RMS bunch size of the provided particle
    distribution

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in units of meters

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the RMS length value in meters.
    """
    s_x = weighted_std(x, weights=w)
    return s_x


def mean_kinetic_energy(px, py, pz, w=None):
    """Calculate the mean kinetic energy of the provided particle distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the mean kinetic energy in non-dimmensional
    units, i.e. [1/(m_e c**2)]
    """
    return np.average(np.sqrt(np.square(px) + np.square(py) + np.square(pz)),
                      weights=w)


def mean_energy(px, py, pz, w=None):
    """Calculate the mean energy of the provided particle distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the mean energy in non-dimmensional units, i.e. [1/(m_e c**2)]
    """
    return np.average(np.sqrt(1 + px**2 + py**2 + pz**2), weights=w)


def rms_energy_spread(px, py, pz, w=None):
    """Calculate the absotule RMS energy spread of the provided particle
    distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    part_ene = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    ene_std = weighted_std(part_ene, weights=w)
    return ene_std


def fwhm_energy_spread(z, px, py, pz, w=None, n_slices=10, len_slice=None):
    """Calculate the absolute FWHM energy spread of the provided particle
    distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    part_ene = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    gamma_hist, z_edges = np.histogram(part_ene, bins=n_slices, weights=w)
    slice_pos = z_edges[1:] - abs(z_edges[1]-z_edges[0])/2
    peak = max(gamma_hist)
    slices_in_fwhm = slice_pos[np.where(gamma_hist >= peak/2)]
    fwhm = max(slices_in_fwhm) - min(slices_in_fwhm)
    return fwhm


def relative_fwhm_energy_spread(z, px, py, pz, w=None, n_slices=10,
                                len_slice=None):
    """Calculate the relative RMS energy spread of the provided particle
    distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the relative energy spread value.
    """
    abs_spread = fwhm_energy_spread(z, px, py, pz, w=w, n_slices=n_slices,
                                    len_slice=len_slice)
    mean_ene = mean_energy(px, py, pz, w)
    rel_spread = abs_spread/mean_ene
    return rel_spread


def relative_rms_energy_spread(px, py, pz, w=None):
    """Calculate the relative RMS energy spread of the provided particle
    distribution

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the relative energy spread value.
    """
    abs_spread = rms_energy_spread(px, py, pz, w)
    mean_ene = mean_energy(px, py, pz, w)
    rel_spread = abs_spread/mean_ene
    return rel_spread


def longitudinal_energy_chirp(z, px, py, pz, w=None):
    """Calculate the longitudinal energy chirp, K, of the provided particle
    distribution in units of m**(-1). It is defined as dE/<E> = K*dz.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the chirp value in units of m^(-1)
    """
    ene = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    mean_ene = np.average(ene, weights=w)
    mean_z = np.average(z, weights=w)
    dE_rel = (ene-mean_ene) / mean_ene
    dz = z - mean_z
    p = np.polyfit(dz, dE_rel, 1)
    K = p[0]
    return K


def rms_relative_correlated_energy_spread(z, px, py, pz, w=None):
    """Calculate the correlated energy spread of the provided particle
    distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    K = longitudinal_energy_chirp(z, px, py, pz, w)
    mean_z = np.average(z, weights=w)
    dz = z - mean_z
    corr_ene = K*dz
    corr_ene_sp = weighted_std(corr_ene, w)
    return corr_ene_sp


def rms_relative_uncorrelated_energy_spread(z, px, py, pz, w=None):
    """Calculate the uncorrelated energy spread of the provided particle
    distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    if len(z) > 1:
        ene = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
        mean_ene = np.average(ene, weights=w)
        mean_z = np.average(z, weights=w)
        dE = ene-mean_ene
        dz = z - mean_z
        p = np.polyfit(dz, dE, 1)
        K = p[0]
        unc_ene = ene - K*dz
        unc_ene_sp = weighted_std(unc_ene, w)/mean_ene
    else:
        unc_ene_sp = 0
    return unc_ene_sp


def normalized_transverse_rms_emittance(x, px, py=None, pz=None, w=None,
                                        disp_corrected=False, corr_order=1):
    """Calculate the normalized transverse RMS emittance without dispersion
    contributions of the particle distribution in a given plane.

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma). Necessary if
        disp_corrected=True.

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma). Necessary if disp_corrected=True.

    w : array or single value
        Statistical weight of the particles.

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    Returns
    -------
    A float with the emmitance value in units of m * rad
    """
    if len(x) > 1:
        if disp_corrected:
            # remove x-gamma correlation
            gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
            gamma_avg = np.average(gamma, weights=w)
            dgamma = (gamma - gamma_avg)/gamma_avg
            x = remove_correlation(dgamma, x, w, corr_order)
        cov_x = np.cov(x, px, aweights=np.abs(w))
        em_x = np.sqrt(np.linalg.det(cov_x.astype(np.float32, copy=False)))
    else:
        em_x = 0
    return em_x


def geometric_transverse_rms_emittance(x, px, py, pz, w=None,
                                       disp_corrected=False, corr_order=1):
    """Calculate the geometric transverse RMS emittance without dispersion
    contributions of the particle distribution in a given plane.

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters
    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)
    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma).
    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).
    w : array or single value
        Statistical weight of the particles.
    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.
    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    Returns
    -------
    A float with the emmitance value in units of m * rad
    """
    gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    gamma_avg = np.average(gamma, weights=w)
    em_x = normalized_transverse_rms_emittance(x, px, py, pz, w,
                                               disp_corrected, corr_order)
    return em_x / gamma_avg


def normalized_transverse_trace_space_rms_emittance(
        x, px, py, pz, w=None, disp_corrected=False, corr_order=1):
    """Calculate the normalized trasnverse trace-space RMS emittance of the
    particle distribution in a given plane.

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma).

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).

    w : array or single value
        Statistical weight of the particles.

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    Returns
    -------
    A float with the emmitance value in units of m * rad
    """
    gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    gamma_avg = np.average(gamma, weights=w)
    em_x = transverse_trace_space_rms_emittance(x, px, py, pz, w,
                                                disp_corrected, corr_order)
    return em_x * gamma_avg


def transverse_trace_space_rms_emittance(x, px, py=None, pz=None, w=None,
                                         disp_corrected=False, corr_order=1):
    """Calculate the trasnverse trace-space RMS emittance of the
    particle distribution in a given plane.

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma). Necessary if
        disp_corrected=True.

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma). Necessary if disp_corrected=True.

    w : array or single value
        Statistical weight of the particles.

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    Returns
    -------
    A float with the emmitance value in units of m * rad
    """
    if len(x) > 1:
        xp = px/pz
        if disp_corrected:
            # remove x-gamma correlation
            gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
            gamma_avg = np.average(gamma, weights=w)
            dgamma = (gamma - gamma_avg)/gamma_avg
            x = remove_correlation(dgamma, x, w, corr_order)
            # remove xp-gamma correlation
            xp = remove_correlation(dgamma, xp, w, corr_order)
        cov_x = np.cov(x, xp, aweights=np.abs(w))
        em_x = np.sqrt(np.linalg.det(cov_x.astype(np.float32, copy=False)))
    else:
        em_x = 0
    return em_x


def longitudinal_rms_emittance(z, px, py, pz, w=None):
    """Calculate the longitudinal RMS emittance of the particle
    distribution in a given plane.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    Returns
    -------
    A float with the emmitance value in units of m
    """
    g = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    cov_l = np.cov(z, g, aweights=np.abs(w))
    em_l = np.sqrt(np.linalg.det(cov_l.astype(np.float32, copy=False)))
    return em_l


def peak_current(z, q, n_slices=10, len_slice=None):
    """Calculate the peak current of the given particle distribution.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    q : array
        Contains the charge of the particles in C

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    The absolute value of the peak current in Ampere.
    """
    current_prof, *_ = current_profile(z, q, n_slices=n_slices,
                                       len_slice=len_slice)
    current_prof = abs(current_prof)
    return max(current_prof)


def fwhm_length(z, q, n_slices=10, len_slice=0.1e-6):
    """Calculate the FWHM length of the given particle distribution.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    q : array
        Contains the charge of the particles in C

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    The FWHM value in metres.
    """
    current_prof, z_edges = current_profile(z, q, n_slices=n_slices,
                                            len_slice=len_slice)
    slice_pos = z_edges[1:] - abs(z_edges[1]-z_edges[0])/2
    current_prof = abs(current_prof)
    i_peak = max(current_prof)
    i_half = i_peak/2
    slices_in_fwhm = slice_pos[np.where(current_prof >= i_half)]
    fwhm = max(slices_in_fwhm) - min(slices_in_fwhm)
    return fwhm


def relative_rms_slice_energy_spread(z, px, py, pz, w=None, n_slices=10,
                                     len_slice=None):
    """Calculate the relative RMS slice energy spread of the provided particle
    distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - An array with the relative energy spread value in each slice.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    - A float with the weigthed average of the slice values.
    """
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    slice_ene_sp = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            px_slice = px[slice_particle_filter]
            py_slice = py[slice_particle_filter]
            pz_slice = pz[slice_particle_filter]
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_ene_sp[i] = relative_rms_energy_spread(px_slice, py_slice,
                                                         pz_slice, w_slice)
            slice_weight[i] = np.sum(w_slice)
    slice_avg = calculate_slice_average(slice_ene_sp, slice_weight)
    return slice_ene_sp, slice_weight, slice_lims, slice_avg


def rms_relative_uncorrelated_slice_energy_spread(z, px, py, pz, w=None,
                                                  n_slices=10, len_slice=None):
    """Calculate the uncorrelated slcie energy spread of the provided particle
    distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - An array with the relative energy spread value in each slice.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    - A float with the weigthed average of the slice values.
    """
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    slice_ene_sp = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            z_slice = z[slice_particle_filter]
            px_slice = px[slice_particle_filter]
            py_slice = py[slice_particle_filter]
            pz_slice = pz[slice_particle_filter]
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_ene_sp[i] = rms_relative_uncorrelated_energy_spread(
                z_slice, px_slice, py_slice, pz_slice, w_slice)
            slice_weight[i] = np.sum(w_slice)
    slice_avg = calculate_slice_average(slice_ene_sp, slice_weight)
    return slice_ene_sp, slice_weight, slice_lims, slice_avg


def normalized_transverse_rms_slice_emittance(
        z, x, px, py=None, pz=None, w=None, disp_corrected=False, corr_order=1,
        n_slices=10, len_slice=None):
    """Calculate the normalized transverse RMS slice emittance of the particle
    distribution in a given plane.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma). Necessary if
        disp_corrected=True.

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma). Necessary if disp_corrected=True.

    w : array or single value
        Statistical weight of the particles.

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - An array with the emmitance value in each slice in units of m * rad.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    - A float with the weigthed average of the slice values.
    """
    if disp_corrected:
        # remove x-gamma correlation
        gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
        gamma_avg = np.average(gamma, weights=w)
        dgamma = (gamma - gamma_avg)/gamma_avg
        x = remove_correlation(dgamma, x, w, corr_order)
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    slice_em = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            x_slice = x[slice_particle_filter]
            px_slice = px[slice_particle_filter]
            # if py is not None:
            #    py_slice = py[slice_particle_filter]
            # else:
            #    py_slice=None
            # if pz is not None:
            #    pz_slice = pz[slice_particle_filter]
            # else:
            #    pz_slice=None
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_em[i] = normalized_transverse_rms_emittance(
                x_slice, px_slice, w=w_slice)
            slice_weight[i] = np.sum(w_slice)
    slice_avg = calculate_slice_average(slice_em, slice_weight)
    return slice_em, slice_weight, slice_lims, slice_avg


def slice_twiss_parameters(
        z, x, px, pz, py=None, w=None, disp_corrected=False, corr_order=1,
        n_slices=10, len_slice=None):
    """Calculate the Twiss parameters for each longitudinal slice of the
    particle distribution in a given plane.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma). Necessary if
        disp_corrected=True.

    w : array or single value
        Statistical weight of the particles.

    disp_corrected : bool
        Whether ot not to correct for dispersion contributions.

    corr_order : int
        Highest order up to which dispersion effects should be corrected.

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - A list with the arrays of the alpha, beta [m] and gamma [m^-1] functions.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    - A list with the weighted average slice values of alpha, beta and gamma.
    """
    if disp_corrected:
        # remove x-gamma correlation
        gamma = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
        gamma_avg = np.average(gamma, weights=w)
        dgamma = (gamma - gamma_avg)/gamma_avg
        x = remove_correlation(dgamma, x, w, corr_order)
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    slice_alpha = np.zeros(n_slices)
    slice_beta = np.zeros(n_slices)
    slice_gamma = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            x_slice = x[slice_particle_filter]
            px_slice = px[slice_particle_filter]
            pz_slice = pz[slice_particle_filter]
            # if py is not None:
            #    py_slice = py[slice_particle_filter]
            # else:
            #    py_slice=None
            # if pz is not None:
            #    pz_slice = pz[slice_particle_filter]
            # else:
            #    pz_slice=None
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_alpha[i], slice_beta[i], slice_gamma[i] = twiss_parameters(
                x_slice, px_slice, pz_slice, w=w_slice)
            slice_weight[i] = np.sum(w_slice)
    slice_twiss_params = [slice_alpha, slice_beta, slice_gamma]
    alpha_avg = calculate_slice_average(slice_alpha, slice_weight)
    beta_avg = calculate_slice_average(slice_beta, slice_weight)
    gamma_avg = calculate_slice_average(slice_gamma, slice_weight)
    slice_avgs = [alpha_avg, beta_avg, gamma_avg]
    return slice_twiss_params, slice_weight, slice_lims, slice_avgs


def energy_profile(z, px, py, pz, w=None, n_slices=10, len_slice=None):
    """Calculate the sliced longitudinal energy profile of the distribution

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - An array with the mean energy value in each slice.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    """
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    slice_ene = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            px_slice = px[slice_particle_filter]
            py_slice = py[slice_particle_filter]
            pz_slice = pz[slice_particle_filter]
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_ene[i] = mean_energy(px_slice, py_slice, pz_slice, w_slice)
            slice_weight[i] = np.sum(w_slice)
    return slice_ene, slice_weight, slice_lims


def current_profile(z, q, n_slices=10, len_slice=None):
    """Calculate the current profile of the given particle distribution.

    Parameters
    ----------
    z : array
        Contains the longitudinal position of the particles in units of meters

    q : array
        Contains the charge of the particles in C

    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.

    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns
    -------
    A tuple containing:
    - An array with the current of each slice in units of A.
    - An array with the slice edges along z.
    """
    slice_lims, n_slices = create_beam_slices(z, n_slices, len_slice)
    sl_len = slice_lims[1] - slice_lims[0]
    charge_hist, z_edges = np.histogram(z, bins=n_slices, weights=q)
    sl_dur = sl_len/ct.c
    current_prof = charge_hist/sl_dur
    return current_prof, z_edges


def energy_spectrum(px, py, pz, w=None, bins=10):
    """Calculate the energy spectrum (histogram) of the given particle
    distribution.

    Parameters
    ----------
    px : array
        Contains the transverse momentum in the x direction of the
        beam particles in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum in the y direction of the
        beam particles in non-dimmensional units (beta*gamma)

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma)

    w : array or single value
        Statistical weight of the particles.

    bins : int
        Number of bins of the histogram.

    Returns
    -------
    A tuple containing:
    - An array with the energy histogram  (normalized to 1).
    - An array with the bin edges of the histogram.
    """
    if w is not None:
        w = np.abs(w)
    gamma = np.sqrt(1 + px**2 + py**2 + pz**2)
    ene_hist, bin_edges = np.histogram(gamma, bins=bins, weights=w)
    ene_hist /= np.max(ene_hist)
    return ene_hist, bin_edges


def general_analysis(x, y, z, px, py, pz, q, n_slices=10, len_slice=None,
                     print_params=False):
    """Quick method to analyze the most relevant beam parameters at once.

    Parameters
    ----------
    x : array
        Contains the transverse position of the particles in the x
        transverse plane in units of meter

    y : array
        Contains the transverse position of the particles in the y
        transverse plane in units of meter

    z : array
        Contains the longitudinal position of the particles in units of meters

    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)

    py : array
        Contains the transverse momentum of the beam particles in the opposite
        plane as as x in non-dimmensional units (beta*gamma).

    pz : array
        Contains the longitudinal momentum of the beam particles in
        non-dimmensional units (beta*gamma).

    q : array
        Charge of the particles in Coulomb.

    len_slice : array
        Length of the longitudinal slices.

    print_params : bool
        If True, the parameters of the analyzed distribution will also be
        printed.

    Returns
    -------
    A tuple containing the centroid position, pointing angle, Twiss parameters,
    bunch length, divergence, energy and the total and slice emittance and
    energy spread.
    """
    q_tot = np.sum(q)
    a_x, b_x, g_x = twiss_parameters(x, px, pz, py, w=q)
    a_y, b_y, g_y = twiss_parameters(y, py, pz, px, w=q)
    ene = mean_energy(px, py, pz, w=q)
    ene_sp = relative_rms_energy_spread(px, py, pz, w=q)
    enespls, sl_w, sl_lim, ene_sp_sl = relative_rms_slice_energy_spread(
        z, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    em_x = normalized_transverse_rms_emittance(x, px, py, pz, w=q)
    em_y = normalized_transverse_rms_emittance(y, py, px, pz, w=q)
    emsx, sl_w,  sl_lim, em_sl_x = normalized_transverse_rms_slice_emittance(
        z, x, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    emsy, sl_w, sl_lim, em_sl_y = normalized_transverse_rms_slice_emittance(
        z, y, py, px, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    i_peak = peak_current(z, q, n_slices=n_slices, len_slice=len_slice)
    z_fwhm = fwhm_length(z, q, n_slices=n_slices, len_slice=len_slice)
    s_z = rms_length(z, w=q)
    s_x = rms_size(x, w=q)
    s_y = rms_size(y, w=q)
    s_px = np.std(px/pz)
    s_py = np.std(py/pz)
    x_centroid = np.average(x, weights=q)
    y_centroid = np.average(y, weights=q)
    z_centroid = np.average(z, weights=q)
    px_centroid = np.average(px, weights=q)
    py_centroid = np.average(py, weights=q)
    theta_x = px_centroid/ene
    theta_y = py_centroid/ene

    params_dict = {
        'x_avg': x_centroid,
        'y_avg': y_centroid,
        'z_avg': z_centroid,
        'theta_x': theta_x,
        'theta_y': theta_y,
        'sigma_x': s_x,
        'sigma_y': s_y,
        'sigma_z': s_z,
        'z_fwhm': z_fwhm,
        'sigma_px': s_px,
        'sigma_py': s_py,
        'alpha_x': a_x,
        'alpha_y': a_y,
        'beta_x': b_x,
        'beta_y': b_y,
        'gamma_x': g_x,
        'gamma_y': g_y,
        'emitt_nx': em_x,
        'emitt_ny': em_y,
        'emitt_nx_sl': em_sl_x,
        'emitt_ny_sl': em_sl_y,
        'ene_avg': ene,
        'rel_ene_sp': ene_sp,
        'rel_ene_sp_sl': ene_sp_sl,
        'i_peak': i_peak,
        'q': q_tot
    }

    if print_params:
        print('Parametes of particle distribution:')
        print('-'*80)
        print('number of particles = {}'.format(len(x)))
        print('q_tot = {:1.2e} C'.format(q_tot))
        print('alpha_x = {:1.2e}, alpha_y = {:1.2e}'.format(a_x, a_y))
        print('beta_x = {:1.2e} m, beta_y = {:1.2e} m'.format(b_x, b_y))
        print('gamma_x = {:1.2e} 1/m, gamma_y = {:1.2e} 1/m'.format(g_x, g_y))
        print('sigma_x = {:1.2e} m, sigma_y = {:1.2e} m'.format(s_x, s_y))
        print('sigma_z = {:1.2e} m (sigma_t = {:1.2e} s)'.format(
            s_z, s_z/ct.c))
        print('z_fwhm = {:1.2e} m (t_fwhm = {:1.2e} s)'.format(
            z_fwhm, z_fwhm/ct.c))
        print('i_peak = {:1.2e} kA'.format(i_peak*1e-3))
        print('norm_emitt_x = {:1.2e} m, norm_emitt_y = {:1.2e} m'.format(
            em_x, em_y))
        print(('norm_emitt_x_sl = {:1.2e} m, '
               + 'norm_emitt_y_sl = {:1.2e} m').format(em_sl_x, em_sl_y))
        print('gamma_avg = {:1.2e} (ene_avg = {:1.2e} MeV)'.format(
            ene, ene*ct.m_e*ct.c**2/ct.e*1e-6))
        print('gamma_spread = {:1.2e} ({:1.2e} %)'.format(ene_sp, ene_sp*100))
        print('gamma_spread_sl = {:1.2e} ({:1.2e} %)'.format(
            ene_sp_sl, ene_sp_sl*100))
        print('-'*80)

    return params_dict
