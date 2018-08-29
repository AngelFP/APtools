"""This module contains methods for reading beam data from different particle
tracking and PIC codes"""

import numpy as np
import scipy.constants as ct
from h5py import File as H5F

from aptools.helper_functions import join_infile_path


def read_csrtrack_data_fmt1(file_path):
    """Reads particle data from CSRtrack in fmt1 format and returns it in the
    unis used by APtools.

    Parameters:
    -----------
    file_path : str
        Path to the file with particle data

    Returns:
    --------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.loadtxt(file_path)
    z = data[1:,0]
    x = data[1:,1]
    y = data[1:,2]
    pz = data[1:,3] / (ct.m_e*ct.c**2/ct.e)
    px = data[1:,4] / (ct.m_e*ct.c**2/ct.e)
    py = data[1:,5] / (ct.m_e*ct.c**2/ct.e)
    q = data[1:,6]
    x[1:] += x[0]
    y[1:] += y[0]
    z[1:] += z[0]
    px[1:] += px[0]
    py[1:] += py[0]
    pz[1:] += pz[0]
    return x, y, z, px, py, pz, q

def read_astra_data(file_path):
    """Reads particle data from ASTRA and returns it in the unis used by
    APtools.

    Parameters:
    -----------
    file_path : str
        Path to the file with particle data

    Returns:
    --------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.loadtxt(file_path)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    px = data[:,3] / (ct.m_e*ct.c**2/ct.e)
    py = data[:,4] / (ct.m_e*ct.c**2/ct.e)
    pz = data[:,5] / (ct.m_e*ct.c**2/ct.e)
    z[1:] += z[0]
    pz[1:] += pz[0]
    q = data[:,7] * 1e-9
    return x, y, z, px, py, pz, q

def read_openpmd_beam(file_path, species_name):
    """Reads particle data from a h5 file following the openPMD standard and
    returns it in the unis used by APtools.

    Parameters:
    -----------
    file_path : str
        Path to the file with particle data

    species_name : str
        Name of the particle species

    Returns:
    --------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    file_content = H5F(file_path)
    # get base path in file
    iteration = list(file_content['/data'].keys())[0]
    base_path = '/data/{}'.format(iteration)
    # get path under which particle data is stored
    particles_path = file_content.attrs['particlesPath'].decode()
    # get species
    beam_species = file_content[
        join_infile_path(base_path, particles_path, species_name) ]
    # get data
    m = beam_species['mass'].attrs['value']
    q = beam_species['charge'].attrs['value']
    x = (beam_species['position/x'][:]
         + beam_species['positionOffset/x'].attrs['value'])
    y = (beam_species['position/y'][:]
         + beam_species['positionOffset/y'].attrs['value'])
    z = (beam_species['position/z'][:]
         + beam_species['positionOffset/z'].attrs['value'])
    px = beam_species['momentum/x'][:] / (m*ct.c)
    py = beam_species['momentum/y'][:] / (m*ct.c)
    pz = beam_species['momentum/z'][:] / (m*ct.c)
    w = beam_species['weighting'][:]
    q *= w
    return x, y, z, px, py, pz, q
