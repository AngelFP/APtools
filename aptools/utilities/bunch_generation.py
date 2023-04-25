""" This module contains methods for generating particle distributions """

import numpy as np
import scipy.constants as ct
from scipy.stats import truncnorm

import aptools.data_handling.reading as dr
import aptools.data_handling.saving as ds
import aptools.data_analysis.beam_diagnostics as bd
import aptools.data_processing.beam_operations as bo


def generate_gaussian_bunch_from_twiss(
        a_x, a_y, b_x, b_y, en_x, en_y, ene, ene_sp, s_t, q_tot, n_part, head_current, 
        x_c=0, y_c=0, z_c=0, lon_profile='gauss', min_len_scale_noise=None,
        sigma_trunc_lon=None, smooth_sigma=None, smooth_trunc=None,
        save_to_file=False, save_to_code='astra',
        save_to_path=None, file_name=None, perform_checks=False):
    """
    Creates a transversely Gaussian particle bunch with the specified Twiss
    parameters.

    Parameters
    ----------
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
        the RMS duration. If lon_profile='flattop' or
        lon_profile='flattop_smoothed', this instead the whole flat-top lenght.

    q_tot: float
        Total bunch charge in C.

    n_part: int
        Total number of particles in the bunch.
    
    head_current: float
        takes a value between 0 and 1.
        if it is 1 all the current is in the head, if it 0 all the current is 
        in the tail, and if it is 0.5 both head and tail have the same current.

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

    smooth_sigma: float
        The sigma of the Gaussian longitudinal smoothing applied to the
        flat-top profile when lon_profile='flattop_smoothed'. Units are in
        seconds.

    smooth_trunc: float
        Number of sigmas after which to truncate the Gaussian smoothing when
        lon_profile='flattop_smoothed'

    save_to_file: bool
        Whether to save the generated distribution to a file.

    save_to_code: string
        (optional) Name of the target code that will use the saved file.
        Possible values are 'csrtrack', 'astra' and 'fbpic'. Required if
        save_to_file=True.

    save_to_path: string
        (optional) Path to the folder where to save the data. Required if
        save_to_file=True.

    file_name: string
        (optional) Name of the file where to store the beam data. Required if
        save_to_file=True.

    perform_checks: bool
        Whether to compute and print the parameters of the generated bunch.

    Returns
    -------
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
    elif lon_profile == 'flattop_smoothed':
        z = _create_flattop_longitudinal_profile_with_smoothing(
            z_c, s_z, n_part, min_len_scale_noise, smooth_sigma, smooth_trunc)
    elif lon_profile == 'rectan_trapezoidal':
        z = _create_rectan_trapezoidal_longitudinal_profile(z_c, s_z, n_part,
                                                            head_current,
                                                            min_len_scale_noise)
    elif lon_profile == 'rectan_trapezoidal_smoothed':
        z = _create_rectan_trapezoidal_longitudinal_profile_smoothed(z_c, 
                                                                     s_z, 
                                                                     n_part,
                                                                     head_current,
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
    if save_to_file:
        print('Saving to file... ', end='')
        ds.save_beam(
            save_to_code, [x, y, z, px, py, pz, q], save_to_path, file_name)
        print('Done.')
    if perform_checks:
        _check_beam_parameters(x, y, z, px, py, pz, q)
    return x, y, z, px, py, pz, q


def generate_from_file_modifying_twiss(
        code_name, file_path, read_kwargs={}, alphax_target=None,
        betax_target=None, alphay_target=None, betay_target=None,
        save_to_file=False, save_to_code='astra', save_to_path=None,
        file_name=None, save_kwargs={}, perform_checks=False):
    """
    Creates a transversely Gaussian particle bunch with the specified Twiss
    parameters.

    Parameters
    ----------
    code_name: str
        Name of the tracking or PIC code of the data to read. Possible values
        are 'csrtrack', 'astra', 'openpmd', 'osiris' and 'hipace'

    file_path: str
        Path of the file containing the data

    read_kwargs: dict
        Dictionary containing optional parameters for the read_beam function.

    save_to_file: bool
        Whether to save the generated distribution to a file.

    save_to_code: string
        (optional) Name of the target code that will use the saved file.
        Possible values are 'csrtrack', 'astra' and 'fbpic'. Required if
        save_to_file=True.

    save_to_path: string
        (optional) Path to the folder where to save the data. Required if
        save_to_file=True.

    file_name: string
        (optional) Name of the file where to store the beam data. Required if
        save_to_file=True.

    save_kwargs: dict
        Dictionary containing optional parameters for the save_beam function.

    perform_checks: bool
        Whether to compute and print the parameters of the generated bunch.

    Returns
    -------
    A tuple with 7 arrays containing the 6D components and charge of the
    modified distribution.

    """
    # Read distribution
    x, y, z, px, py, pz, q = dr.read_beam(code_name, file_path, **read_kwargs)
    # Modify Twiss parameters
    x, y, z, px, py, pz, q = bo.modify_twiss_parameters_all_beam(
        [x, y, z, px, py, pz, q], alphax_target=alphax_target,
        betax_target=betax_target, alphay_target=alphay_target,
        betay_target=betay_target)
    # Save to file
    if save_to_file:
        print('Saving to file... ', end='')
        ds.save_beam(
            save_to_code, [x, y, z, px, py, pz, q], save_to_path, file_name,
            **save_kwargs)
        print('Done.')
        # Perform checks
    if perform_checks:
        _check_beam_parameters(x, y, z, px, py, pz, q)
    return x, y, z, px, py, pz, q


def _create_gaussian_longitudinal_profile(z_c, s_z, n_part, sigma_trunc_lon,
                                          min_len_scale_noise):
    """ Creates a Gaussian longitudinal profile """
    # Make sure number of particles is an integer
    n_part = int(n_part)
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


def _create_flattop_longitudinal_profile(z_c, length, n_part,
                                         min_len_scale_noise):
    """ Creates a flattop longitudinal profile """
    # Make sure number of particles is an integer
    n_part = int(n_part)
    if min_len_scale_noise is None:
        z = np.random.uniform(z_c-length/2, z_c+length/2, n_part)
    else:
        n_slices = int(np.round(length/(min_len_scale_noise)))
        part_per_slice = np.round(np.ones(n_slices)*n_part/n_slices)
        part_per_slice = part_per_slice.astype(int)
        slice_edges = np.linspace(z_c-length/2, z_c+length/2, n_slices+1)
        z = _create_smooth_z_array(part_per_slice, slice_edges)
    return z


def _create_flattop_longitudinal_profile_with_smoothing(z_c, length, n_part,
                                                        min_len_scale_noise,
                                                        smooth_sigma,
                                                        smooth_trunc):
    """ Creates a flattop longitudinal profile with gaussian smoothing at the
    head and tail"""
    # Number of particles
    smooth_sigma = ct.c*smooth_sigma
    n_plat = n_part * length/(length+np.sqrt(2*np.pi*smooth_sigma**2))
    n_smooth = n_part * (np.sqrt(2*np.pi*smooth_sigma**2)
                         / (length+np.sqrt(2*np.pi*smooth_sigma**2)))
    # Create flattop and gaussian profiles
    z_plat = _create_flattop_longitudinal_profile(length/2, length, n_plat,
                                                  min_len_scale_noise)
    z_smooth = _create_gaussian_longitudinal_profile(0, smooth_sigma, n_smooth,
                                                     smooth_trunc,
                                                     min_len_scale_noise)
    # Concatenate both profiles
    z = np.concatenate((z_smooth[np.where(z_smooth <= 0)],
                        z_plat,
                        z_smooth[np.where(z_smooth > 0)] + length))
    # Center distribution around desired position
    z = z - length/2 + z_c
    return z


def _create_rectan_trapezoidal_longitudinal_profile(z_c, length, n_part, 
                                                    head_current,
                                                    min_len_scale_noise):
    """ Creates a rectangular trapezoidal longitudinal profile """
    
    # Make sure number of particles is an integer
    n_part = int(n_part)
    if min_len_scale_noise is None:
        a = z_c-length/2
        b = z_c+length/2
        
        if head_current == 0.5:
            head_current += 1e-12

        hmax = 2/(b-a)
        h1 = head_current * hmax

        h2 = 2/(b-a) - h1
        if h2 < 0:
            raise ValueError("h2 = 2/(b-a) - h1 < 0!")

        z = rectan_trapezoid(n_part,a=a,b=b,h1=h1,h2=h2)
    else:
        raise NotImplementedError('Noise reduction not implemented for `trapezoidal` profile.')
    return z


def _create_rectan_trapezoidal_longitudinal_profile_smoothed(z_c, length, 
                                                             n_part, head_current,
                                                             min_len_scale_noise,
                                                             left_weight=0.05, 
                                                             right_weight=0.05):
    
    """ Creates a rectangular trapezoidal smoothed longitudinal profile """
    
    # Make sure number of particles is an integer
    n_part = int(n_part)
    if min_len_scale_noise is None:
        z = rectan_trapezoid_smoothed(n_part=n_part,a=z_c-length/2,b=z_c+length/2,
                                      h1=head_current,left_weight=left_weight,
                                     right_weight=right_weight)
    else:
        raise NotImplementedError('Noise reduction not implemented for `trapezoidal` profile.')
    return z
    

def rectan_trapezoid(n_part, a, b, h1, h2):
    """
    Creates a longitudinal rectangular trapezoid particle bunch with the specified
    parameters.
    Parameters
    ----------
    n_part : float
        Number of particles in the beam
       
             __       
          __/  |
       __/     |
     /         |
    |          |
    |h1        |h2 
    |__________|
    a          b
    
    a : float
        start position of the trapezoid
    b : float
        end position of the trapezoid
    h1 : float
        first height. h1 can only have a value between 0 and maximum height: 2/(b-a)
    h2 : float
        second height

    ----------
    Returns
    
    x : array with size [n_part]
        distribution of random variables with rectangular trapezoidal shape
        
    """
    n_part = int(n_part)
    y = np.random.uniform(0,1,n_part)
    
    x = np.zeros(len(y))
    
    for i in range(len(y)):
        if y[i] >= 0 and y[i] <= 1:
            n = h1
            m = (h2-h1)/(2*(b-a))
            D = n**2 + 4 * m * y[i]
            x[i] = (-n+np.sqrt(D))/(2*m) + a
    return x


def half_gaussian(n_part, mu, peak, part):
    """ 
    Creates half gaussian distribution 
    
                                 
                           /|     peak|\
              left -----> / |         | \ <---- right gaussian distribution
          gaussian       /  |         |  \
      distribution      /   |         |   \
                                    mu
    
    n_part : float
        Number of particles in the distribution
    mu : float
        Mean of the distribution, if it would be whole gaussian distribution
    peak : float
        Height of the distribution
    part : string
        it can be only "left" or "right".
        indicates which part of gaussian distribution you want to create.
    ----------
    Returns
    
    x : array with size [n_part]
        distribution of random variables with half gaussian shape
    
    """

    if peak == 0:
        peak += 1e-12
    
    # Since we use all the particles and translate them either to the left or to the right, 
    # if the number doubles on the left/right, 
    # then the height is halved (or sigma is doubled)
    
    sigma = 1 / (peak / 2 * np.sqrt(2 * np.pi))
    
    x = np.random.normal(loc=mu, scale=sigma, size=n_part)
    if part == 'right':
        idx = x < mu
    elif part == 'left':
        idx = x > mu
    else:
        raise ValueError('Wrong part')
    x[idx] = 2 * mu - x[idx]
    return x


def rectan_trapezoid_smoothed(n_part, a, b, h1, left_weight, right_weight):
    """
    Creates a longitudinal rectangular trapezoid particle bunch with smoothed sides.
    Parameters
    ----------
    n_part : float
        Number of particles in the beam
       
                             ______       
                          __/      |\
                      ___/         | \
   gaussian         /|             |  \
   distribution    / |  trapezoial |   \  gaussian distribution
  0.05% n_part--> /  |distribution |    \ <--0.05% n_part
                 /__ |__0.9n_part__|_____\
                    a              b

    a : float
        start position of the trapezoid
    b : float
        end position of the trapezoid
    h1 : float
        first height
        % of the maximum height
    h2 : float
        second height
        
    left_weight : float
        % from the whole area which goes to the left half-gaussian distribution
    
    right_weight : float
        % from the whole area which goes to the right half-gaussian distribution
        
    plot: bool
        if True, then plot PDF
    ----------
    Returns
    
    x : array with size [n_part]
        distribution of random variables with rectangular trapezoidal smoothed shape 
    """
    
    trapez_weight = 1 - (left_weight + right_weight)
    n_part = int(n_part)
    
    if h1 == 0.5:
        h1 += 1e-12
    elif h1 < 0.01 or h1 > 0.99:
        raise ValueError('Please, put head_current in range [0.01,0.99]')

    hmax = 2/(b-a) * trapez_weight
    h1 = h1 * hmax 

    h2 = hmax - h1
    if h2 < 0:
        raise ValueError("h2 = 2/(b-a) - h1 < 0!")
        
    p = np.random.rand(n_part)
    x = np.zeros_like(p)
    
    idx_left = p < left_weight
    idx_right = p > (1 - right_weight)
    idx = np.logical_not(np.logical_or(idx_left, idx_right))
    
    if trapez_weight != 0:
        x[idx] = rectan_trapezoid(np.count_nonzero(idx), a, b, 
                                  h1 / trapez_weight, 
                                  h2 / trapez_weight,plot=False)
    if left_weight != 0:
        x[idx_left] = half_gaussian(np.count_nonzero(idx_left), mu=a, 
                                    peak=h1 / left_weight, part='left')
    if right_weight != 0:
        x[idx_right] = half_gaussian(np.count_nonzero(idx_right), mu=b, 
                                     peak=h2 / right_weight, part='right')
    return x


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
    b_x = beam_params['beta_x']
    b_y = beam_params['beta_y']
    a_x = beam_params['alpha_x']
    a_y = beam_params['alpha_y']
    s_x = beam_params['sigma_x']
    s_y = beam_params['sigma_y']
    s_z = beam_params['sigma_z']
    em_x = beam_params['emitt_nx']
    em_y = beam_params['emitt_ny']
    ene = beam_params['ene_avg']
    ene_sp = beam_params['rel_ene_sp']
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
