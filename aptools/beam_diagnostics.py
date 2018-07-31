"""This module defines methods calculating beam properties"""

import scipy.constants as ct
import numpy as np

from .helper_functions import weighted_std

def twiss_parameters(x, px, pz, w=1):
    """Calculate the alpha and beta functions of the beam in a certain
    transverse plane

    Parameters:
    -----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters
    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)
    pz : array
        Contains the longitudonal momentum of the beam particles in
        non-dimmensional units (beta*gamma)
    w : array or single value
        Statistical weight of the particles.

    Returns:
    --------
    A tuple with the value of the alpha and beta functions [m]
    """
    xp = px/pz
    cov_x = np.cov(x, xp, aweights=np.abs(w))
    em_x = np.sqrt(np.linalg.det(cov_x))
    b_x = cov_x[0, 0]/em_x
    a_x = -cov_x[0, 1]/em_x
    return (a_x, b_x)

def rms_length(z, w=1):
    """Calculate the RMS bunch length of the provided particle
    distribution

    Parameters:
    -----------
    z : array
        Contains the longitudinal position of the particles in units of meters
    w : array or single value
        Statistical weight of the particles.

    Returns:
    --------
    A float with the RMS length value in meters.
    """
    s_z = weighted_std(z, weights=w)
    return s_z

def rms_size(x, w=1):
    """Calculate the RMS bunch size of the provided particle
    distribution

    Parameters:
    -----------
    x : array
        Contains the transverse position of the particles in units of meters
    w : array or single value
        Statistical weight of the particles.

    Returns:
    --------
    A float with the RMS length value in meters.
    """
    s_x = weighted_std(x, weights=w)
    return s_x

def mean_kinetic_energy(px, py, pz, w=1):
    """Calculate the mean kinetic energy of the provided particle distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the mean kinetic energy in non-dimmensional
    units, i.e. [1/(m_e c**2)]
    """
    return np.average(np.sqrt(np.square(px) + np.square(py) + np.square(pz)),
                      weights=np.abs(w))

def mean_energy(px, py, pz, w=1):
    """Calculate the mean energy of the provided particle distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the mean energy in non-dimmensional units, i.e. [1/(m_e c**2)]
    """
    kin_ene = mean_kinetic_energy(px, py, pz, w)
    return np.average(1 + np.sqrt(np.square(kin_ene)))

def rms_energy_spread(px, py, pz, w=1):
    """Calculate the absotule RMS energy spread of the provided particle
    distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    part_ene = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    ene_std = weighted_std(part_ene, weights=w)
    return ene_std

def relative_rms_energy_spread(px, py, pz, w=1):
    """Calculate the relative RMS energy spread of the provided particle
    distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    abs_spread = rms_energy_spread(px, py, pz, w)
    mean_ene = mean_energy(px, py, pz, w)
    rel_spread = abs_spread/mean_ene
    return rel_spread

def longitudinal_energy_chirp(z, px, py, pz, w=1):
    """Calculate the longitudinal energy chirp, K, of the provided particle
    distribution in units of m**(-1). It is defined as dE/<E> = K*dz.

    Parameters:
    -----------
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

    Returns:
    --------
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

def rms_correlated_energy_spread(z, px, py, pz, w=1):
    """Calculate the correlated energy spread of the provided particle
    distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the energy spread value in non-dimmensional units,
    i.e. [1/(m_e c**2)]
    """
    K = longitudinal_energy_chirp(z, px, py, pz, w)
    mean_z = np.average(z, weights=w)
    dz = z - mean_z
    corr_ene = K*dz
    corr_ene_sp = weighted_std(corr_ene, w)
    return corr_ene_sp

def normalized_transverse_rms_emittance(x, px, w=1):
    """Calculate the normalized transverse RMS emittance of the particle
    distribution in a given plane.

    Parameters:
    -----------
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters
    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)
    w : array or single value
        Statistical weight of the particles.

    Returns:
    --------
    A float with the emmitance value in units of m * rad
    """
    cov_x = np.cov(x, px, aweights=np.abs(w))
    em_x = np.sqrt(np.linalg.det(cov_x))
    return em_x

def longitudinal_rms_emittance(z, px, py, pz, w=1):
    """Calculate the longitudinal RMS emittance of the particle
    distribution in a given plane.

    Parameters:
    -----------
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

    Returns:
    --------
    A float with the emmitance value in units of m
    """
    g = np.sqrt(1 + np.square(px) + np.square(py) + np.square(pz))
    cov_l = np.cov(z, g, aweights=np.abs(w))
    em_l = np.sqrt(np.linalg.det(cov_l))
    return em_l

def relative_rms_slice_energy_spread(z, px, py, pz, w=1, n_slices=10,
                                     len_slice=None):
    """Calculate the relative RMS slice energy spread of the provided particle
    distribution

    Parameters:
    -----------
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

    Returns:
    --------
    A tuple containing:
    - An array with the energy spread value in each slice. Values are in
    non-dimmensional units, i.e. [1/(m_e c**2)]
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    """
    max_z = np.max(z)
    min_z = np.min(z)
    if len_slice is None:
        slice_lims = np.linspace(min_z, max_z, n_slices+1)
    else:
        slice_lims = np.arange(min_z, max_z, len_slice)
        slice_lims = np.append(slice_lims, max_z)
        n_slices = len(slice_lims)-1
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
            slice_ene_sp[i] = relative_rms_energy_spread(px_slice, py_slice,
                                                         pz_slice, w_slice)
            slice_weight[i] = np.sum(w_slice)
    return slice_ene_sp, slice_weight, slice_lims

def normalized_transverse_rms_slice_emittance(z, x, px, w=1, n_slices=10,
                                     len_slice=None):
    """Calculate the normalized transverse RMS slice emittance of the particle
    distribution in a given plane.

    Parameters:
    -----------
    z : array
        Contains the longitudinal position of the particles in units of meters
    x : array
        Contains the transverse position of the particles in one of the
        transverse planes in units of meters
    px : array
        Contains the transverse momentum of the beam particles in the same
        plane as x in non-dimmensional units (beta*gamma)
    w : array or single value
        Statistical weight of the particles.
    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.
    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns:
    --------
    A tuple containing:
    - An array with the emmitance value in each slice in units of m * rad.
    - An array with the statistical weight of each slice.
    - An array with the slice edges.
    """
    max_z = np.max(z)
    min_z = np.min(z)
    if len_slice is None:
        slice_lims = np.linspace(min_z, max_z, n_slices+1)
    else:
        slice_lims = np.arange(min_z, max_z, len_slice)
        slice_lims = np.append(slice_lims, max_z)
        n_slices = len(slice_lims)-1
    slice_em = np.zeros(n_slices)
    slice_weight = np.zeros(n_slices)
    for i in np.arange(0, n_slices):
        a = slice_lims[i]
        b = slice_lims[i+1]
        slice_particle_filter = (z > a) & (z <= b)
        if slice_particle_filter.any():
            x_slice = x[slice_particle_filter]
            px_slice = px[slice_particle_filter]
            if hasattr(w, '__iter__'):
                w_slice = w[slice_particle_filter]
            else:
                w_slice = w
            slice_em[i] = normalized_transverse_rms_emittance(
                x_slice, px_slice, w_slice)
            slice_weight[i] = np.sum(w_slice)
    return slice_em, slice_weight, slice_lims

def current_profile(z, q, n_slices=10, len_slice=None):
    """Calculate the current profile of the given particle distribution.

    Parameters:
    -----------
    z : array
        Contains the longitudinal position of the particles in units of meters
    q : array
        Contains the charge of the particles in C
    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.
    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns:
    --------
    A tuple containing:
    - An array with the current of each slice in units of A.
    - An array with the slice edges along z.
    """
    max_z = np.max(z)
    min_z = np.min(z)
    if len_slice is not None:
        n_slices = int((max_z-min_z)/len_slice)
    adj_slice_len = (max_z-min_z)/n_slices
    print('Slice length = {} m'.format(adj_slice_len))
    charge_hist, z_edges = np.histogram(z, bins=n_slices, weights=q)
    sl_dur = adj_slice_len/ct.c
    current_prof = charge_hist/sl_dur
    return current_prof, z_edges
