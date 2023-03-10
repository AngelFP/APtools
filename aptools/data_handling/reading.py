"""This module contains methods for reading beam data from different particle
tracking and PIC codes"""

import numpy as np
import scipy.constants as ct
from h5py import File as H5File
from deprecated import deprecated

from aptools.helper_functions import join_infile_path, reposition_bunch
from aptools.plasma_accel.general_equations import plasma_skin_depth
from aptools.data_processing.beam_filtering import filter_beam


@deprecated(
    version="0.2.0",
    reason=("This method is replaced by those in the new "
            "`particle_distributions.read` module.")
)
def read_beam(code_name, file_path, reposition=False,
              avg_pos=[None, None, None], avg_mom=[None, None, None],
              filter_min=[None, None, None, None, None, None, None],
              filter_max=[None, None, None, None, None, None, None],
              **kwargs):
    """Reads particle data from the specified code.

    Parameters
    ----------
    code_name : str
        Name of the tracking or PIC code of the data to read. Possible values
        are 'csrtrack', 'astra', 'openpmd', 'osiris', 'hipace' and 'fbpic'.

    file_path : str
        Path of the file containing the data

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

    filter_min, filter_max : list
        List of length 7 with the minimum and maximum value for each particle
        component, ordered as [x, y, z, pz, py, pz, q]. Particles outside the
        given range will be filtered out. For values which are 'None', no
        filtering is performed.

    Other Parameters
    ----------------
    **kwargs
        This method takes additional keyword parameters that might be needed
        for some data readers. Possible parameters are 'species_name' and
        'plasma_dens'.
    """
    read_beam_from = {'csrtrack': read_csrtrack_data_fmt1,
                      'astra': read_astra_data,
                      'openpmd': read_openpmd_beam,
                      'osiris': read_osiris_beam,
                      'hipace': read_hipace_beam,
                      'fbpic': read_fbpic_input_beam}
    x, y, z, px, py, pz, q = read_beam_from[code_name](file_path, **kwargs)
    if reposition:
        reposition_bunch([x, y, z, px, py, pz, q], avg_pos+avg_mom)
    if any(filter_min + filter_max):
        x, y, z, px, py, pz, q = filter_beam(
            np.array([x, y, z, px, py, pz, q]), filter_min, filter_max)
    return x, y, z, px, py, pz, q


def read_csrtrack_data_fmt1(file_path):
    """Reads particle data from CSRtrack in fmt1 format and returns it in the
    unis used by APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.genfromtxt(file_path)
    z = data[1:, 0]
    x = data[1:, 1]
    y = data[1:, 2]
    pz = data[1:, 3] / (ct.m_e*ct.c**2/ct.e)
    px = data[1:, 4] / (ct.m_e*ct.c**2/ct.e)
    py = data[1:, 5] / (ct.m_e*ct.c**2/ct.e)
    q = data[1:, 6]
    x[1:] += x[0]
    y[1:] += y[0]
    z[1:] += z[0]
    px[1:] += px[0]
    py[1:] += py[0]
    pz[1:] += pz[0]
    return x, y, z, px, py, pz, q


def read_astra_data(file_path, remove_non_standard=True):
    """Reads particle data from ASTRA and returns it in the unis used by
    APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    remove_non_standard : bool
        Determines whether non-standard particles (those with a status flag
        other than 5) should be removed from the read data.

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.genfromtxt(file_path)
    status_flag = data[:, 9]
    if remove_non_standard:
        data = data[np.where(status_flag == 5)]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    px = data[:, 3] / (ct.m_e*ct.c**2/ct.e)
    py = data[:, 4] / (ct.m_e*ct.c**2/ct.e)
    pz = data[:, 5] / (ct.m_e*ct.c**2/ct.e)
    z[1:] += z[0]
    pz[1:] += pz[0]
    q = data[:, 7] * 1e-9
    return x, y, z, px, py, pz, q


def read_openpmd_beam(file_path, species_name=None):
    """Reads particle data from a h5 file following the openPMD standard and
    returns it in the unis used by APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    species_name : str, Optional
        Name of the particle species. Optional if only one particle species
        is available in the openpmd file.

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    file_content = H5File(file_path, mode='r')
    # get base path in file
    iteration = list(file_content['/data'].keys())[0]
    base_path = '/data/{}'.format(iteration)
    # get path under which particle data is stored
    particles_path = file_content.attrs['particlesPath'].decode()
    # list available species
    available_species = list(
        file_content[join_infile_path(base_path, particles_path)])
    if species_name is None:
        if len(available_species) == 1:
            species_name = available_species[0]
        else:
            raise ValueError(
                'More than one particle species is available. '
                'Please specify a `species_name`. '
                'Available species are: ' + str(available_species))
    # get species
    beam_species = file_content[
        join_infile_path(base_path, particles_path, species_name)]
    # get data
    mass = beam_species['mass']
    charge = beam_species['charge']
    position = beam_species['position']
    position_off = beam_species['positionOffset']
    momentum = beam_species['momentum']
    m = mass.attrs['value'] * mass.attrs['unitSI']
    q = charge.attrs['value'] * charge.attrs['unitSI']
    x = (position['x'][:] * position['x'].attrs['unitSI'] +
         position_off['x'].attrs['value'] * position_off['x'].attrs['unitSI'])
    y = (position['y'][:] * position['y'].attrs['unitSI'] +
         position_off['y'].attrs['value'] * position_off['y'].attrs['unitSI'])
    z = (position['z'][:] * position['z'].attrs['unitSI'] +
         position_off['z'].attrs['value'] * position_off['z'].attrs['unitSI'])
    px = momentum['x'][:] * momentum['x'].attrs['unitSI'] / (m*ct.c)
    py = momentum['y'][:] * momentum['y'].attrs['unitSI'] / (m*ct.c)
    pz = momentum['z'][:] * momentum['z'].attrs['unitSI'] / (m*ct.c)
    w = beam_species['weighting'][:]
    q *= w
    return x, y, z, px, py, pz, q


def read_hipace_beam(file_path, plasma_dens):
    """Reads particle data from an HiPACE paricle file and returns it in the
    unis used by APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    plasma_dens : float
        Plasma density in units od cm^{-3} used to convert the beam data to
        non-normalized units

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    s_d = plasma_skin_depth(plasma_dens)
    file_content = H5File(file_path, mode='r')
    # sim parameters
    n_cells = file_content.attrs['NX']
    sim_size = (file_content.attrs['XMAX'] - file_content.attrs['XMIN'])
    cell_vol = np.prod(sim_size/n_cells)
    q_norm = cell_vol * plasma_dens * 1e6 * s_d**3 * ct.e
    # get data
    q = np.array(file_content.get('q')) * q_norm
    x = np.array(file_content.get('x2')) * s_d
    y = np.array(file_content.get('x3')) * s_d
    z = np.array(file_content.get('x1')) * s_d
    px = np.array(file_content.get('p2'))
    py = np.array(file_content.get('p3'))
    pz = np.array(file_content.get('p1'))
    return x, y, z, px, py, pz, q


def read_osiris_beam(file_path, plasma_dens):
    """Reads particle data from an OSIRIS paricle file and returns it in the
    unis used by APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    plasma_dens : float
        Plasma density in units od cm^{-3} used to convert the beam data to
        non-normalized units

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    s_d = plasma_skin_depth(plasma_dens)
    file_content = H5File(file_path, mode='r')
    # get data
    q = np.array(file_content.get('q')) * ct.e
    x = np.array(file_content.get('x2')) * s_d
    y = np.array(file_content.get('x3')) * s_d
    z = np.array(file_content.get('x1')) * s_d
    px = np.array(file_content.get('p2'))
    py = np.array(file_content.get('p3'))
    pz = np.array(file_content.get('p1'))
    return x, y, z, px, py, pz, q


def read_fbpic_input_beam(file_path, q_tot):
    """Reads particle data from an FBPIC input beam file

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    q_tot: float
        Total beam charge.
    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.genfromtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    px = data[:, 3]
    py = data[:, 4]
    pz = data[:, 5]
    q = np.ones(len(x))*q_tot/len(x)
    return x, y, z, px, py, pz, q
