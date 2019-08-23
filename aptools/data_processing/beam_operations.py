""" This module contains methods that perform operations on particle
distributions"""

import numpy as np

from aptools.data_analysis.beam_diagnostics import twiss_parameters


def modify_twiss_parameters_all_beam(beam_data, betax_target=None,
                                     alphax_target=None, gammax_target=None,
                                     betay_target=None, alphay_target=None,
                                     gammay_target=None):
    """
    Modifies the transverse distribution of a particle bunch in both transverse
    planes so that it has the specified Twiss parameters.

    Parameters
    ----------
    beam_data: iterable
        List, tuple or array containing the beam data arrays as
        [x, y, z, px, py, pz, q].

    betax_target: float
        Target beta in the x-plane (horizontal) of the resulting distribution.
        Not necessary if alphax_target and gammax_target are already provided.

    alphax_target: float
        Target alpha in the x-plane (horizontal) of the resulting distribution.
        Not necessary if betax_target and gammax_target are already provided.

    gammax_target: float
        Target gamma in the x-plane (horizontal) of the resulting distribution.
        Not necessary if betax_target and alphax_target are already provided.

    betay_target: float
        Target beta in the y-plane (vertical) of the resulting distribution.
        Not necessary if alphay_target and gammay_target are already provided.

    alphay_target: float
        Target alpha in the y-plane (vertical) of the resulting distribution.
        Not necessary if betay_target and gammay_target are already provided.

    gammay_target: float
        Target gamma in the y-plane (vertical) of the resulting distribution.
        Not necessary if betay_target and alphay_target are already provided.

    Returns
    -------
    A tuple with 7 arrays containing the 6D components and charge of the
    modified distribution.

    """
    x, y, z, px, py, pz, q = beam_data
    x, px = modify_twiss_parameters(x, px, pz, q, betax_target, alphax_target,
                                    gammax_target)
    y, py = modify_twiss_parameters(y, py, pz, q, betay_target, alphay_target,
                                    gammay_target)
    return x, y, z, px, py, pz, q


def modify_twiss_parameters(x, px, pz, weights=None, beta_target=None,
                            alpha_target=None, gamma_target=None):
    """
    Modifies the transverse distribution of a particle bunch so that it has
    the specified Twiss parameters.

    Parameters
    ----------
    x: array
        Transverse position of the particles in meters.

    px: array
        Transverse momentum of the particles in the same plane as the x-array.
        The momentum should be in non-dimmensional units (beta*gamma).

    pz: array
        Longitudinal momentum of the beam particles in non-dimmensional units
        (beta*gamma).

    weights: array
        Statistical weight of the particles.

    beta_target: float
        Target beta of the resulting distribution. Not necessary if
        alpha_target and gamma_target are already provided.

    alpha_target: float
        Target alpha of the resulting distribution. Not necessary if
        beta_target and gamma_target are already provided.

    gamma_target: float
        Target gamma of the resulting distribution. Not necessary if
        beta_target and alpha_target are already provided.

    Returns
    -------
    A tuple with the new x and px arrays satisfying the target Twiss
    parameters.

    """
    # Check target Twiss parameters
    if beta_target is not None:
        bx = beta_target
        if alpha_target is not None:
            ax = alpha_target
            gx = (1+ax**2)/bx
        elif gamma_target is not None:
            gx = gamma_target
            ax = np.sqrt(bx*gx-1)
        else:
            print('Not enough parameters, please specify also a target alpha'
                  ' or gamma value.')
            return x, px
    elif alpha_target is not None:
        ax = alpha_target
        if gamma_target is not None:
            gx = gamma_target
            bx = (1+ax**2)/gx
        else:
            print('Not enough parameters, please specify also a target beta or'
                  ' gamma value.')
            return x, px
    elif gamma_target is not None:
        print('Not enough parameters, please specify also a target beta or'
              ' alpha value.')
        return x, px
    else:
        print('No target Twiss parameters specified. Please provide at least'
              ' two.')
        return x, px
    # Calculate initial Twiss parameters
    ax_0, bx_0, gx_0 = twiss_parameters(x, px, pz, w=weights)
    # Calculate transform matrix assuming M12=0
    M11 = np.sqrt(bx/bx_0)
    M22 = np.sqrt(bx_0/bx)
    M21 = (M22*ax_0 - ax/M11)/bx_0
    M = np.zeros((2, 2))
    M[0, 0] = M11
    M[1, 0] = M21
    M[1, 1] = M22
    # Apply transform matrix
    xp = px/pz
    x_new, xp_new = M.dot(np.vstack((x, xp)))
    px_new = xp_new*pz
    return x_new, px_new
