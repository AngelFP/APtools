""" This module contains methods that perform operations on particle
distributions"""

import numpy as np

from aptools.data_analysis.beam_diagnostics import (twiss_parameters,
                                                    general_analysis)


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
    M = np.zeros((2,2))
    M[0,0] = M11
    M[1,0] = M21
    M[1,1] = M22
    # Apply transform matrix
    xp = px/pz
    x_new, xp_new = M.dot(np.vstack((x, xp)))
    px_new = xp_new*pz
    ax_n, bx_n, gx_n = twiss_parameters(x_new, px_new, pz, w=weights)
    print('New beam parameters:')
    print('-'*80)
    print('alpha_x = {:1.2e}'.format(ax_n))
    print('beta_x = {:1.2e}'.format(bx_n))
    print('gamma_x = {:1.2e}'.format(gx_n))
    return x_new, px_new
