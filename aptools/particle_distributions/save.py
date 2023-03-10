"""Defines methods for saving particle distributions to file."""

from typing import Optional

import numpy as np
import scipy.constants as ct
from openpmd_api import (Series, Access, Dataset, Mesh_Record_Component,
                         Unit_Dimension)

from aptools import __version__
from .particle_distribution import ParticleDistribution


SCALAR = Mesh_Record_Component.SCALAR


def save_distribution(
    distribution: ParticleDistribution,
    file_path: str,
    data_format: str,
    **kwargs
) -> None:
    """Save particle distribution to file in the specified format.

    Parameters
    ----------
    distribution : ParticleDistribution
        The particle distribution to save.
    file_path : str
        Path to the file in which to save the data.
    data_format : str
        Internal format of the data.  Possible values
        are 'astra', 'csrtrack' and 'openpmd'.

    Other Parameters
    ----------------
    **kwargs
        Additional parameters to be passed to the particle savers.
    """
    _savers[data_format](distribution, file_path, **kwargs)


def save_to_astra(
    distribution: ParticleDistribution,
    file_path: str
) -> None:
    """Save particle distribution to text file in ASTRA format.

    Parameters
    ----------
    distribution : ParticleDistribution
        The particle distribution to save.
    file_path : str
        Path to the file in which to save the data.
    """

    # Get beam data
    m_species = distribution.m_species
    q_species = distribution.q_species
    x_orig = distribution.x
    y_orig = distribution.y
    z_orig = distribution.z
    px_orig = distribution.px * m_species * ct.c**2 / ct.e  # eV/c
    py_orig = distribution.py * m_species * ct.c**2 / ct.e  # eV/c
    pz_orig = distribution.pz * m_species * ct.c**2 / ct.e  # eV/c
    q_orig = distribution.q * 1e9  # nC
    w = distribution.w

    # Determine particle index (type of species).
    if m_species == ct.m_e and q_species == -ct.e:
        index = 1
    elif m_species == ct.m_e and q_species == ct.e:
        index = 2
    elif m_species == ct.m_p and q_species == ct.e:
        index = 3
    else:
        raise ValueError(
            'Only electrons, positrons and protons are supported when saving '
            'to ASTRA.')

    # Create arrays
    x = np.zeros(q_orig.size + 1)
    y = np.zeros(q_orig.size + 1)
    z = np.zeros(q_orig.size + 1)
    px = np.zeros(q_orig.size + 1)
    py = np.zeros(q_orig.size + 1)
    pz = np.zeros(q_orig.size + 1)
    q = np.zeros(q_orig.size + 1)

    # Reference particle
    x[0] = np.average(x_orig, weights=w)
    y[0] = np.average(y_orig, weights=w)
    z[0] = np.average(z_orig, weights=w)
    px[0] = np.average(px_orig, weights=w)
    py[0] = np.average(py_orig, weights=w)
    pz[0] = np.average(pz_orig, weights=w)
    q[0] = np.sum(q_orig) / len(q_orig)

    # Put relative to reference particle
    x[1::] = x_orig
    y[1::] = y_orig
    z[1::] = z_orig - z[0]
    px[1::] = px_orig
    py[1::] = py_orig
    pz[1::] = pz_orig - pz[0]
    q[1::] = q_orig
    t = z / ct.c

    # Add flags and indices
    ind = np.ones(q.size) * index
    flag = np.ones(q.size) * 5

    # Save to file
    data = np.column_stack((x, y, z, px, py, pz, t, q, ind, flag))
    np.savetxt(
        file_path,
        data,
        '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %i %i'
    )


def save_to_csrtrack(
    distribution: ParticleDistribution,
    file_path: str
) -> None:
    """Save particle distribution to text file using the CSRtrack fmt1 format.

    Parameters
    ----------
    distribution : ParticleDistribution
        The particle distribution to save.
    file_path : str
        Path to the file in which to save the data.
    """

    # Get beam data.
    x_orig = distribution.x
    y_orig = distribution.y
    z_orig = distribution.z
    px_orig = distribution.px * ct.m_e*ct.c**2/ct.e
    py_orig = distribution.py * ct.m_e*ct.c**2/ct.e
    pz_orig = distribution.pz * ct.m_e*ct.c**2/ct.e
    q_orig = distribution.q

    # Create arrays.
    x = np.zeros(q_orig.size+2)
    y = np.zeros(q_orig.size+2)
    z = np.zeros(q_orig.size+2)
    px = np.zeros(q_orig.size+2)
    py = np.zeros(q_orig.size+2)
    pz = np.zeros(q_orig.size+2)
    q = np.zeros(q_orig.size+2)

    # Reference particle.
    x[1] = np.average(x_orig, weights=q_orig)
    y[1] = np.average(y_orig, weights=q_orig)
    z[1] = np.average(z_orig, weights=q_orig)
    px[1] = np.average(px_orig, weights=q_orig)
    py[1] = np.average(py_orig, weights=q_orig)
    pz[1] = np.average(pz_orig, weights=q_orig)
    q[1] = sum(q_orig)/len(q_orig)

    # Relative coordinates.
    x[2::] = x_orig - x[1]
    y[2::] = y_orig - y[1]
    z[2::] = z_orig - z[1]
    px[2::] = px_orig - px[1]
    py[2::] = py_orig - py[1]
    pz[2::] = pz_orig - pz[1]
    q[2::] = q_orig

    # Save to file.
    data = np.column_stack((z, x, y, pz, px, py, q))
    if not file_path.endswith('.fmt1'):
        file_path += '.fmt1'
    np.savetxt(
        file_path,
        data,
        '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e'
    )


def save_to_openpmd(
    distribution: ParticleDistribution,
    file_path: str,
    species_name: Optional[str] = 'particle_distribution'
) -> None:
    """
    Save particle distribution to an HDF5 file following the openPMD standard.

    Parameters
    ----------
    distribution : ParticleDistribution
        The particle distribution to save.
    file_path : str
        Path to the file in which to save the data.
    species_name : str
        Optional. Name under which the particle species should be stored.

    """
    # Get beam data
    x = np.ascontiguousarray(distribution.x)
    y = np.ascontiguousarray(distribution.y)
    z = np.ascontiguousarray(distribution.z)
    px = np.ascontiguousarray(distribution.px)
    py = np.ascontiguousarray(distribution.py)
    pz = np.ascontiguousarray(distribution.pz)
    w = np.ascontiguousarray(distribution.w)
    q_species = distribution.q_species
    m_species = distribution.m_species

    # Save to file
    if not file_path.endswith('.h5'):
        file_path += '.h5'
    opmd_series = Series(file_path, Access.create)

    # Set basic attributes.
    opmd_series.set_software('APtools', __version__)
    opmd_series.set_particles_path('particles')

    # Create iteration
    it = opmd_series.iterations[0]

    # Create particles species.
    particles = it.particles[species_name]

    # Create additional necessary arrays and constants.
    px = px * m_species * ct.c
    py = py * m_species * ct.c
    pz = pz * m_species * ct.c

    # Generate datasets.
    d_x = Dataset(x.dtype, extent=x.shape)
    d_y = Dataset(y.dtype, extent=y.shape)
    d_z = Dataset(z.dtype, extent=z.shape)
    d_px = Dataset(px.dtype, extent=px.shape)
    d_py = Dataset(py.dtype, extent=py.shape)
    d_pz = Dataset(pz.dtype, extent=pz.shape)
    d_w = Dataset(w.dtype, extent=w.shape)
    d_q = Dataset(np.dtype('float64'), extent=[1])
    d_m = Dataset(np.dtype('float64'), extent=[1])
    d_xoff = Dataset(np.dtype('float64'), extent=[1])
    d_yoff = Dataset(np.dtype('float64'), extent=[1])
    d_zoff = Dataset(np.dtype('float64'), extent=[1])

    # Record data.
    particles['position']['x'].reset_dataset(d_x)
    particles['position']['y'].reset_dataset(d_y)
    particles['position']['z'].reset_dataset(d_z)
    particles['positionOffset']['x'].reset_dataset(d_xoff)
    particles['positionOffset']['y'].reset_dataset(d_yoff)
    particles['positionOffset']['z'].reset_dataset(d_zoff)
    particles['momentum']['x'].reset_dataset(d_px)
    particles['momentum']['y'].reset_dataset(d_py)
    particles['momentum']['z'].reset_dataset(d_pz)
    particles['weighting'][SCALAR].reset_dataset(d_w)
    particles['charge'][SCALAR].reset_dataset(d_q)
    particles['mass'][SCALAR].reset_dataset(d_m)

    # Prepare for writting.
    particles['position']['x'].store_chunk(x)
    particles['position']['y'].store_chunk(y)
    particles['position']['z'].store_chunk(z)
    particles['positionOffset']['x'].make_constant(0.)
    particles['positionOffset']['y'].make_constant(0.)
    particles['positionOffset']['z'].make_constant(0.)
    particles['momentum']['x'].store_chunk(px)
    particles['momentum']['y'].store_chunk(py)
    particles['momentum']['z'].store_chunk(pz)
    particles['weighting'][SCALAR].store_chunk(w)
    particles['charge'][SCALAR].make_constant(q_species)
    particles['mass'][SCALAR].make_constant(m_species)

    # Set units.
    particles['position'].unit_dimension = {Unit_Dimension.L: 1}
    particles['positionOffset'].unit_dimension = {Unit_Dimension.L: 1}
    particles['momentum'].unit_dimension = {
        Unit_Dimension.L: 1,
        Unit_Dimension.M: 1,
        Unit_Dimension.T: -1,
        }
    particles['charge'].unit_dimension = {
        Unit_Dimension.T: 1,
        Unit_Dimension.I: 1,
        }
    particles['mass'].unit_dimension = {Unit_Dimension.M: 1}

    # Set weighting attributes.
    particles['position'].set_attribute('macroWeighted', np.uint32(0))
    particles['positionOffset'].set_attribute(
        'macroWeighted', np.uint32(0))
    particles['momentum'].set_attribute('macroWeighted', np.uint32(0))
    particles['weighting'][SCALAR].set_attribute(
        'macroWeighted', np.uint32(1))
    particles['charge'][SCALAR].set_attribute(
        'macroWeighted', np.uint32(0))
    particles['mass'][SCALAR].set_attribute('macroWeighted', np.uint32(0))
    particles['position'].set_attribute('weightingPower', 0.)
    particles['positionOffset'].set_attribute('weightingPower', 0.)
    particles['momentum'].set_attribute('weightingPower', 1.)
    particles['weighting'][SCALAR].set_attribute('weightingPower', 1.)
    particles['charge'][SCALAR].set_attribute('weightingPower', 1.)
    particles['mass'][SCALAR].set_attribute('weightingPower', 1.)

    # Flush data.
    opmd_series.flush()


_savers = {
    'astra': save_to_astra,
    'csrtrack': save_to_csrtrack,
    'openpmd': save_to_openpmd
}
