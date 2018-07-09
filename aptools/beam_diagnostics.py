"""This module defines methods calculating beam properties"""

import scipy.constants as ct
import numpy as np

from .helper_functions import weighted_std

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

def correlated_energy_spread(z, px, py, pz, w=1):
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
