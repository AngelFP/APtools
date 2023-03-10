"""This module contains methods for saving beam data for different particle
tracking and PIC codes"""

from os import path

import numpy as np
import scipy.constants as ct
from openpmd_api import (Series, Access, Dataset, Mesh_Record_Component,
                         Unit_Dimension)
from deprecated import deprecated

from aptools.helper_functions import reposition_bunch, get_particle_subset
from aptools import __version__


SCALAR = Mesh_Record_Component.SCALAR


@deprecated(
    version="0.2.0",
    reason=("This method is replaced by those in the new "
            "`particle_distributions.save` module.")
)
def save_beam(code_name, beam_data, folder_path, file_name, reposition=False,
              avg_pos=[None, None, None], avg_mom=[None, None, None],
              n_part=None, **kwargs):
    """Converts particle data from one code to another.

    Parameters
    ----------
    code_name : str
        Name of the target tracking or PIC code. Possible values are
        'csrtrack', 'astra', 'fbpic' and 'openpmd'.

    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save, without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.
    """
    save_beam_for = {'csrtrack': save_for_csrtrack_fmt1,
                     'astra': save_for_astra,
                     'fbpic': save_for_fbpic,
                     'openpmd': save_to_openpmd_file}
    save_beam_for[code_name](beam_data, folder_path, file_name, reposition,
                             avg_pos, avg_mom, n_part, **kwargs)


def save_for_csrtrack_fmt1(beam_data, folder_path, file_name, reposition=False,
                           avg_pos=[None, None, None],
                           avg_mom=[None, None, None], n_part=None):
    """Saves particle data for CSRtrack in fmt1 format.

    Parameters
    ----------
    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.
    """
    # Perform repositioning of original distribution
    if reposition:
        reposition_bunch(beam_data, avg_pos+avg_mom)

    # Create subset of n_part
    if n_part is not None:
        beam_data = get_particle_subset(
            beam_data, n_part, preserve_charge=True)

    # Get beam data
    x_orig = beam_data[0]
    y_orig = beam_data[1]
    xi_orig = beam_data[2]
    px_orig = beam_data[3]*ct.m_e*ct.c**2/ct.e
    py_orig = beam_data[4]*ct.m_e*ct.c**2/ct.e
    pz_orig = beam_data[5]*ct.m_e*ct.c**2/ct.e
    q_orig = beam_data[6]

    # Create arrays
    x = np.zeros(q_orig.size+2)
    y = np.zeros(q_orig.size+2)
    xi = np.zeros(q_orig.size+2)
    px = np.zeros(q_orig.size+2)
    py = np.zeros(q_orig.size+2)
    pz = np.zeros(q_orig.size+2)
    q = np.zeros(q_orig.size+2)

    # Reference particle
    x[1] = np.average(x_orig, weights=q_orig)
    y[1] = np.average(y_orig, weights=q_orig)
    xi[1] = np.average(xi_orig, weights=q_orig)
    px[1] = np.average(px_orig, weights=q_orig)
    py[1] = np.average(py_orig, weights=q_orig)
    pz[1] = np.average(pz_orig, weights=q_orig)
    q[1] = sum(q_orig)/len(q_orig)

    # Relative coordinates
    x[2::] = x_orig - x[1]
    y[2::] = y_orig - y[1]
    xi[2::] = xi_orig - xi[1]
    px[2::] = px_orig - px[1]
    py[2::] = py_orig - py[1]
    pz[2::] = pz_orig - pz[1]
    q[2::] = q_orig
    # Save to file
    data = np.column_stack((xi, x, y, pz, px, py, q))
    file_name += '.fmt1'
    np.savetxt(path.join(folder_path, file_name), data,
               '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e')


def save_for_astra(beam_data, folder_path, file_name, reposition=False,
                   avg_pos=[None, None, None], avg_mom=[None, None, None],
                   n_part=None):
    """Saves particle data in ASTRA format.

    Parameters
    ----------
    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.
    """
    # Perform repositioning of original distribution
    if reposition:
        reposition_bunch(beam_data, avg_pos+avg_mom)

    # Create subset of n_part
    if n_part is not None:
        beam_data = get_particle_subset(
            beam_data, n_part, preserve_charge=True)

    # Get beam data
    x_orig = beam_data[0]
    y_orig = beam_data[1]
    xi_orig = beam_data[2]
    px_orig = beam_data[3]*ct.m_e*ct.c**2/ct.e
    py_orig = beam_data[4]*ct.m_e*ct.c**2/ct.e
    pz_orig = beam_data[5]*ct.m_e*ct.c**2/ct.e
    q_orig = beam_data[6]*1e9  # nC

    # Create arrays
    x = np.zeros(q_orig.size+1)
    y = np.zeros(q_orig.size+1)
    xi = np.zeros(q_orig.size+1)
    px = np.zeros(q_orig.size+1)
    py = np.zeros(q_orig.size+1)
    pz = np.zeros(q_orig.size+1)
    q = np.zeros(q_orig.size+1)

    # Reference particle
    x[0] = np.average(x_orig, weights=q_orig)
    y[0] = np.average(y_orig, weights=q_orig)
    xi[0] = np.average(xi_orig, weights=q_orig)
    px[0] = np.average(px_orig, weights=q_orig)
    py[0] = np.average(py_orig, weights=q_orig)
    pz[0] = np.average(pz_orig, weights=q_orig)
    q[0] = sum(q_orig)/len(q_orig)

    # Put relative to reference particle
    x[1::] = x_orig
    y[1::] = y_orig
    xi[1::] = xi_orig - xi[0]
    px[1::] = px_orig
    py[1::] = py_orig
    pz[1::] = pz_orig - pz[0]
    q[1::] = q_orig
    t = xi/ct.c

    # Add flags and indices
    ind = np.ones(q.size)
    flag = np.ones(q.size)*5

    # Save to file
    data = np.column_stack((x, y, xi, px, py, pz, t, q, ind, flag))
    file_name += '.txt'
    np.savetxt(
        path.join(folder_path, file_name), data,
        '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %i %i')


def save_for_fbpic(beam_data, folder_path, file_name, reposition=False,
                   avg_pos=[None, None, None], avg_mom=[None, None, None],
                   n_part=None):
    """Saves particle data in in a format that can be read by FBPIC.

    Parameters
    ----------
    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.
    """
    # Perform repositioning of original distribution
    if reposition:
        reposition_bunch(beam_data, avg_pos+avg_mom)

    # Create subset of n_part
    if n_part is not None:
        beam_data = get_particle_subset(
            beam_data, n_part, preserve_charge=True)

    # Get beam data
    x = beam_data[0]
    y = beam_data[1]
    xi = beam_data[2]
    px = beam_data[3]
    py = beam_data[4]
    pz = beam_data[5]

    # Save to file
    data = np.column_stack((x, y, xi, px, py, pz))
    file_name += '.txt'
    np.savetxt(path.join(folder_path, file_name), data,
               '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e')


def save_to_openpmd_file(
        beam_data, folder_path, file_name, reposition=False,
        avg_pos=[None, None, None], avg_mom=[None, None, None], n_part=None,
        species_name='particle_beam'):
    """
    Saves particle data to an HDF5 file following the openPMD standard.

    Parameters
    ----------
    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.

    species_name : str
        Optional. Name under which the particle species should be stored.

    """

    # Perform repositioning of original distribution
    if reposition:
        reposition_bunch(beam_data, avg_pos+avg_mom)

    # Create subset of n_part
    if n_part is not None:
        beam_data = get_particle_subset(
            beam_data, n_part, preserve_charge=True)

    # Get beam data
    x = np.ascontiguousarray(beam_data[0])
    y = np.ascontiguousarray(beam_data[1])
    z = np.ascontiguousarray(beam_data[2])
    px = np.ascontiguousarray(beam_data[3])
    py = np.ascontiguousarray(beam_data[4])
    pz = np.ascontiguousarray(beam_data[5])
    q = np.ascontiguousarray(beam_data[6])

    # Save to file
    file_path = path.join(folder_path, file_name)
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
    w = np.abs(q) / ct.e
    m = ct.m_e
    q = -ct.e
    px = px * m * ct.c
    py = py * m * ct.c
    pz = pz * m * ct.c

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
    particles['charge'][SCALAR].make_constant(q)
    particles['mass'][SCALAR].make_constant(m)

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
