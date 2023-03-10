"""Defines methods for reading particle distributions from file."""

from typing import Optional

import numpy as np
import scipy.constants as ct
from h5py import File

from aptools.helper_functions import join_infile_path
from .particle_distribution import ParticleDistribution


def read_distribution(
    file_path: str,
    data_format: str,
    **kwargs
) -> ParticleDistribution:
    """Read particle distribution from file.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data
    data_format : str
        Internal format of the data.  Possible values
        are 'astra', 'csrtrack' and 'openpmd'.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters to be passed to the particle readers.

    Returns
    -------
    ParticleDistribution
        The particle distribution.
    """
    return _readers[data_format](file_path, **kwargs)


def read_from_astra(
    file_path: str,
    remove_non_standard: bool = True
) -> ParticleDistribution:
    """Read particle distribution from ASTRA.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data
    remove_non_standard : bool
        Determines whether non-standard particles (those with a status flag
        other than 5) should be removed from the read data.

    Returns
    -------
    ParticleDistribution
        The particle distribution.
    """
    # Read data.
    data = np.genfromtxt(file_path)

    # Get status flag and remove non-standard particles, if needed.
    status_flag = data[:, 9]
    if remove_non_standard:
        data = data[np.where(status_flag == 5)]

    # Extract phase space and particle index.
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    px = data[:, 3]
    py = data[:, 4]
    pz = data[:, 5]
    q = data[:, 7] * 1e-9
    index = data[:, 8]

    # Apply reference particle.
    z[1:] += z[0]
    pz[1:] += pz[0]

    # Determine charge and mass of particle species.
    assert index[0] in [1, 2, 3], (
        'Only electrons, positrons and protons are supported when reading '
        'ASTRA distributions')
    q_species = ct.e * (-1 if index[0] in [1, 3] else 1)
    m_species = ct.m_e if index[0] in [1, 2] else ct.m_p

    # Convert momentum to normalized units (beta * gamma).
    px /= m_species * ct.c**2 / ct.e
    py /= m_species * ct.c**2 / ct.e
    pz /= m_species * ct.c**2 / ct.e

    # Get particle weights.
    w = q / q_species

    # Return distribution.
    return ParticleDistribution(x, y, z, px, py, pz, w, q_species, m_species)


def read_from_csrtrack(
    file_path: str,
) -> ParticleDistribution:
    """Read particle distribution from CSRtrack.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    Returns
    -------
    ParticleDistribution
        The particle distribution.
    """
    # Read data.
    data = np.genfromtxt(file_path)

    # Extract phase space,
    z = data[1:, 0]
    x = data[1:, 1]
    y = data[1:, 2]
    pz = data[1:, 3] / (ct.m_e*ct.c**2/ct.e)
    px = data[1:, 4] / (ct.m_e*ct.c**2/ct.e)
    py = data[1:, 5] / (ct.m_e*ct.c**2/ct.e)
    q = data[1:, 6]

    # Apply reference particle.
    x[1:] += x[0]
    y[1:] += y[0]
    z[1:] += z[0]
    px[1:] += px[0]
    py[1:] += py[0]
    pz[1:] += pz[0]

    # Determine charge and mass of particle species (only electrons in
    # CSRtrack).
    q_species = -ct.e
    m_species = ct.m_e

    # Get particle weights.
    w = q / q_species

    # Return distribution.
    return ParticleDistribution(x, y, z, px, py, pz, w, q_species, m_species)


def read_from_openpmd(
    file_path: str,
    species_name: Optional[str] = None
) -> ParticleDistribution:
    """Read particle distribution from ASTRA.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data
    species_name : str, Optional
        Name of the particle species. Optional if only one particle species
        is available in the openpmd file.

    Returns
    -------
    ParticleDistribution
        The particle distribution.
    """
    # Open file.
    file_content = File(file_path, mode='r')

    # Get base path in file.
    iteration = list(file_content['/data'].keys())[0]
    base_path = '/data/{}'.format(iteration)

    # Get path under which particle data is stored.
    particles_path = file_content.attrs['particlesPath'].decode()

    # Get list of available species.
    available_species = list(
        file_content[join_infile_path(base_path, particles_path)])
    assert len(available_species) > 0, (
        "No particle species found in '{}'.".format(file_path))

    # It not specified, read first species.
    if species_name is None:
        species_name = available_species[0]

    # Get species.
    beam_species = file_content[
        join_infile_path(base_path, particles_path, species_name)]

    # Get phase space and attributes.
    mass = beam_species['mass']
    charge = beam_species['charge']
    position = beam_species['position']
    position_off = beam_species['positionOffset']
    momentum = beam_species['momentum']
    m_species = mass.attrs['value'] * mass.attrs['unitSI']
    q_species = charge.attrs['value'] * charge.attrs['unitSI']
    x = (position['x'][:] * position['x'].attrs['unitSI'] +
         position_off['x'].attrs['value'] * position_off['x'].attrs['unitSI'])
    y = (position['y'][:] * position['y'].attrs['unitSI'] +
         position_off['y'].attrs['value'] * position_off['y'].attrs['unitSI'])
    z = (position['z'][:] * position['z'].attrs['unitSI'] +
         position_off['z'].attrs['value'] * position_off['z'].attrs['unitSI'])
    px = momentum['x'][:] * momentum['x'].attrs['unitSI'] / (m_species*ct.c)
    py = momentum['y'][:] * momentum['y'].attrs['unitSI'] / (m_species*ct.c)
    pz = momentum['z'][:] * momentum['z'].attrs['unitSI'] / (m_species*ct.c)
    w = beam_species['weighting'][:]

    # Return distribution.
    return ParticleDistribution(x, y, z, px, py, pz, w, q_species, m_species)


_readers = {
    'astra': read_from_astra,
    'csrtrack': read_from_csrtrack,
    'openpmd': read_from_openpmd
}
