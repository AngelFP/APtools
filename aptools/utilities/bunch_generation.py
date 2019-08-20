""" This module contains methods for generating particle distributions """

import numpy as np
import scipy.constants as ct
from scipy.stats import truncnorm

import aptools.data_handling.saving as ds
import aptools.data_analysis.beam_diagnostics as bd


def generate_gaussian_bunch_from_twiss(
        a_x, a_y, b_x, b_y, en_x, en_y, ene, ene_sp, s_t, q_tot, n_part, x_c=0,
        y_c=0, z_c=0, lon_profile='gauss', min_len_scale_noise=None,
        sigma_trunc_lon=None, save_to_file=None, save_to_code='astra',
        perform_checks=True):
    """
    Creates a transversely Gaussian particle bunch with the specified Twiss
    parameters.

    Parameters:
    -----------
    a_x : float
        Alpha parameter in the x-plane.

    a_y : float
        Alpha parameter in the y-plane.

    b_x : float
        Beta parameter in the x-plane in units of m.

    b_y : float
        Beta parameter in the y-plane in units of m.

    en_x : float
        Normalized trace-space emittance in the x-plane in units of m*rad.

    en_y : float
        Normalized trace-space emittance in the y-plane in units of m*rad.

    ene: float
        Mean bunch energy in non-dimmensional units (beta*gamma).

    ene_sp: float
        Relative energy spread in %.

    s_t: float
        Bunch duration in seconds. If lon_profile='gauss', this corresponds to
        the RMS duration. If lon_profile='flattop', this instead the whole
        flat-top lenght.

    q_tot: float
        Total bunch charge in C.

    n_part: int
        Total number of particles in the bunch.

    x_c: float
        Central bunch position in the x-plane in units of m.

    y_c: float
        Central bunch position in the y-plane in units of m.

    z_c: float
        Central bunch position in the z-plane in units of m.

    lon_profile: string
        Longitudonal profile of the bunch. Possible values are 'gauss' and
        'flattop'.

    min_len_scale_noise: float
        (optional) If specified, a different algorithm to generate a less noisy
        longitudinal profile is used. This algorithm creates a profile that is
        smooth for a longitudinal binning of the bunch with
        bin lengths >= min_len_scale_noise

    sigma_trunc_lon: float
        (optional) If specified, it truncates the longitudinal distribution of
        the bunch between [z_c-sigma_trunc_lon*s_z, z_c+sigma_trunc_lon*s_z].
        Only used when lon_profile = 'gauss' and required if
        min_len_scale_noise is specified.

    save_to_file: string
        (optional) If specified, the generated distribution will be saved to
        the path specified with this variable.

    save_to_code: string
        (optional) Name of the target code that will use the saved file.
        Possible values are 'csrtrack', 'astra' and 'fbpic'. Required if
        save_to_file is specified.

    perform_checks: bool
        Whether to compute and print the parameters of the generated bunch.

    Returns:
    --------
    The 6D components and charge of the bunch in 7 arrays.

    """
    print('Generating particle distribution... ', end='')
    # Calculate necessary values
    n_part = int(n_part)
    ene_sp = ene_sp/100
    ene_sp_abs = ene_sp*ene
    s_z = s_t*ct.c
    em_x = en_x/ene
    em_y = en_y/ene
    g_x = (1+a_x**2)/b_x
    g_y = (1+a_y**2)/b_y
    s_x = np.sqrt(em_x*b_x)
    s_y = np.sqrt(em_y*b_y)
    s_xp = np.sqrt(em_x*g_x)
    s_yp = np.sqrt(em_y*g_y)
    p_x = -a_x*em_x/(s_x*s_xp)
    p_y = -a_y*em_y/(s_y*s_yp)
    # Create longitudinal distributions
    if lon_profile == 'gauss':
        z = _create_gaussian_longitudinal_profile(z_c, s_z, n_part,
                                                  sigma_trunc_lon,
                                                  min_len_scale_noise)
    elif lon_profile == 'flattop':
        z = _create_flattop_longitudinal_profile(z_c, s_z, n_part,
                                                 min_len_scale_noise)
    # Define again n_part in case it changed when crealing long. profile
    n_part = len(z)
    pz = np.random.normal(ene, ene_sp_abs, n_part)
    # Create normalized gaussian distributions
    u_x = np.random.standard_normal(n_part)
    v_x = np.random.standard_normal(n_part)
    u_y = np.random.standard_normal(n_part)
    v_y = np.random.standard_normal(n_part)
    # Calculate transverse particle distributions
    x = s_x*u_x
    xp = s_xp*(p_x*u_x + np.sqrt(1-np.square(p_x))*v_x)
    y = s_y*u_y
    yp = s_yp*(p_y*u_y + np.sqrt(1-np.square(p_y))*v_y)
    # Change from slope to momentum
    px = xp*pz
    py = yp*pz
    # Charge
    q = np.ones(n_part)*(q_tot/n_part)
    print('Done.')
    # Save to file
    if save_to_file is not None:
        print('Saving to file... ', end='')
        ds.save_beam(save_to_code, [x, y, z, px, py, pz, q], save_to_file)
        print('Done.')
    if perform_checks:
        _check_beam_parameters(x, y, z, px, py, pz, q)
    return x, y, z, px, py, pz, q


def _create_gaussian_longitudinal_profile(z_c, s_z, n_part, sigma_trunc_lon,
                                          min_len_scale_noise):
    """ Creates a Gaussian longitudinal profile """
    if min_len_scale_noise is None:
        if sigma_trunc_lon is not None:
            z = truncnorm.rvs(-sigma_trunc_lon, sigma_trunc_lon, loc=z_c,
                              scale=s_z, size=n_part)
        else:
            z = np.random.normal(z_c, s_z, n_part)
    else:
        tot_len = 2*sigma_trunc_lon*s_z
        n_slices = int(np.round(tot_len/(min_len_scale_noise)))
        part_per_slice = 2*sigma_trunc_lon*n_part/n_slices * truncnorm.pdf(
            np.linspace(-sigma_trunc_lon, sigma_trunc_lon, n_slices),
            -sigma_trunc_lon, sigma_trunc_lon)
        part_per_slice = part_per_slice.astype(int)
        slice_edges = np.linspace(z_c-sigma_trunc_lon*s_z,
                                  z_c+sigma_trunc_lon*s_z,
                                  n_slices+1)
        z = _create_smooth_z_array(part_per_slice, slice_edges)
    return z


def _create_flattop_longitudinal_profile(z_c, s_z, n_part,
                                         min_len_scale_noise):
    """ Creates a flattop longitudinal profile """
    if min_len_scale_noise is None:
        z = np.random.uniform(z_c-s_z/2, z_c+s_z/2, n_part)
    else:
        n_slices = int(np.round(s_z/(min_len_scale_noise)))
        part_per_slice = np.round(np.ones(n_slices)*n_part/n_slices)
        part_per_slice = part_per_slice.astype(int)
        slice_edges = np.linspace(z_c-s_z/2, z_c+s_z/2, n_slices+1)
        z = _create_smooth_z_array(part_per_slice, slice_edges)
    return z


def _create_smooth_z_array(part_per_slice, slice_edges):
    """ Creates the z array of the distribution when forced to be smooth """
    z = np.array([])
    for i, part in enumerate(part_per_slice):
        z_sl = np.linspace(slice_edges[i], slice_edges[i+1], part+2)[1:-1]
        np.random.shuffle(z_sl)
        z = np.concatenate((z, z_sl))
    return z


def _check_beam_parameters(x, y, z, px, py, pz, q):
    """ Analyzes and prints the parameters of the generated distribution """
    print('Performing checks... ', end='')
    beam_params = bd.general_analysis(x, y, z, px, py, pz, q)
    (x_centroid, y_centroid, theta_x, theta_y, b_x, b_y, a_x, a_y, g_x, g_y,
     s_x, s_y, s_z, s_px, s_py, em_x, em_y, ene, ene_sp, em_sl_x, em_sl_y,
     ene_sp_sl) = beam_params
    print('Done.')
    print('Generated beam with:')
    print('-'*80)
    print('alpha_x = {:1.2e}, alpha_y = {:1.2e}'.format(a_x, a_y))
    print('beta_x = {:1.2e} m, beta_y = {:1.2e} m'.format(b_x, b_y))
    print('sigma_x = {:1.2e} m, sigma_y = {:1.2e} m'.format(s_x, s_y))
    print('sigma_z = {:1.2e} m (sigma_t = {:1.2e} s)'.format(s_z, s_z/ct.c))
    print('norm_emitt_x = {:1.2e} m, norm_emitt_y = {:1.2e} m'.format(em_x,
                                                                      em_y))
    print('gamma_avg = {:1.2e}, gamma_spread = {:1.2e}'.format(ene, ene_sp))
    print('-'*80)
