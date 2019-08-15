"""This module contains methods for saving beam data for different particle
tracking and PIC codes"""

from os import path

import numpy as np
import scipy.constants as ct

from aptools.helper_functions import reposition_bunch


def save_beam(code_name, beam_data, folder_path, file_name, reposition=False,
              avg_pos=[None, None, None], avg_mom=[None, None, None],
              n_part=None):
    """Converts particle data from one code to another.

    Parameters:
    -----------
    code_name : str
        Name of the target tracking or PIC code. Possible values are
        'csrtrack', 'astra' and 'fbpic'

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
                     'fbpic': save_for_fbpic}
    save_beam_for[code_name](beam_data, folder_path, file_name, reposition,
                             avg_pos, avg_mom, n_part)


def save_for_csrtrack_fmt1(beam_data, folder_path, file_name, reposition=False,
                           avg_pos=[None, None, None],
                           avg_mom=[None, None, None], n_part=None):
    """Saves particle data for CSRtrack in fmt1 format.

    Parameters:
    -----------
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

    # Get beam data
    x_orig = beam_data[0]
    y_orig = beam_data[1]
    xi_orig = beam_data[2]
    px_orig = beam_data[3]*ct.m_e*ct.c**2/ct.e
    py_orig = beam_data[4]*ct.m_e*ct.c**2/ct.e
    pz_orig = beam_data[5]*ct.m_e*ct.c**2/ct.e
    q_orig = beam_data[6]

    # Create subset of n_part
    if (n_part is not None and n_part < len(q_orig)):
        q_tot = np.sum(q_orig)
        q_part = q_tot/n_part
        i = np.arange(len(q_orig), dtype=np.int32)
        i = np.random.choice(i, size=n_part)
        x_orig = x_orig[i]
        y_orig = y_orig[i]
        xi_orig = xi_orig[i]
        px_orig = px_orig[i]
        py_orig = py_orig[i]
        pz_orig = pz_orig[i]
        q_orig = np.ones(x_orig.size)*q_part

    # Create arrays
    x = np.zeros(q_orig.size+2)
    y = np.zeros(q_orig.size+2)
    xi = np.zeros(q_orig.size+2)
    px = np.zeros(q_orig.size+2)
    py = np.zeros(q_orig.size+2)
    pz = np.zeros(q_orig.size+2)
    pz = np.zeros(q_orig.size+2)
    q = np.zeros(q_orig.size+2)

    # Reference particle
    x[1] = np.average(x_orig, weights=q_orig)
    y[1] = np.average(y_orig, weights=q_orig)
    xi[1] = np.average(xi_orig, weights=q_orig)
    px[1] = np.average(px_orig, weights=q_orig)
    py[1] = np.average(py_orig, weights=q_orig)
    pz[1] = np.average(pz_orig, weights=q_orig)
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

    Parameters:
    -----------
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

    # Get beam data
    x_orig = beam_data[0]
    y_orig = beam_data[1]
    xi_orig = beam_data[2]
    px_orig = beam_data[3]*ct.m_e*ct.c**2/ct.e
    py_orig = beam_data[4]*ct.m_e*ct.c**2/ct.e
    pz_orig = beam_data[5]*ct.m_e*ct.c**2/ct.e
    q_orig = beam_data[6]*1e9  # nC

    # Create subset of n_part
    if (n_part is not None and n_part < len(q_orig)):
        q_tot = np.sum(q_orig)
        q_part = q_tot/n_part
        i = np.arange(len(q_orig))
        i = np.random.choice(i, size=n_part)
        x_orig = x_orig[i]
        y_orig = y_orig[i]
        xi_orig = xi_orig[i]
        px_orig = px_orig[i]
        py_orig = py_orig[i]
        pz_orig = pz_orig[i]
        q_orig = np.ones(x_orig.size)*q_part

    # Create arrays
    x = np.zeros(q_orig.size+1)
    y = np.zeros(q_orig.size+1)
    xi = np.zeros(q_orig.size+1)
    px = np.zeros(q_orig.size+1)
    py = np.zeros(q_orig.size+1)
    pz = np.zeros(q_orig.size+1)
    pz = np.zeros(q_orig.size+1)
    q = np.zeros(q_orig.size+1)

    # Reference particle
    x[0] = np.average(x_orig, weights=q_orig)
    y[0] = np.average(y_orig, weights=q_orig)
    xi[0] = np.average(xi_orig, weights=q_orig)
    px[0] = np.average(px_orig, weights=q_orig)
    py[0] = np.average(py_orig, weights=q_orig)
    pz[0] = np.average(pz_orig, weights=q_orig)
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

    Parameters:
    -----------
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

    # Get beam data
    x = beam_data[0]
    y = beam_data[1]
    xi = beam_data[2]
    px = beam_data[3]
    py = beam_data[4]
    pz = beam_data[5]
    q = beam_data[6]

    # Create subset of n_part
    if (n_part is not None and n_part < len(q)):
        q_tot = np.sum(q)
        q_part = q_tot/n_part
        i = np.arange(len(q))
        i = np.random.choice(i, size=n_part)
        x = x[i]
        y = y[i]
        xi = xi[i]
        px = px[i]
        py = py[i]
        pz = pz[i]
        q = np.ones(x.size)*q_part

    # Save to file
    data = np.column_stack((x, y, xi, px, py, pz))
    file_name += '.txt'
    np.savetxt(path.join(folder_path, file_name), data,
               '%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e')
