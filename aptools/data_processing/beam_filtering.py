"""This module defines methods for filtering and selecting beam data"""

import numpy as np

from aptools.helper_functions import weighted_std


def filter_beam(beam_matrix, min_range, max_range):
    """Filter the beam particles keeping only those within a given range.

    Parameters
    ----------
    beam_matrix : array
        M x N matrix containing all M components of the N particles.

    min_range : array
        Array of size M with the minimum value for each given component. For
        values which are 'None' no filtering is performed.

    max_range : array
        Array of size M with the maximum value for each given component. For
        values which are 'None' no filtering is performed.

    Returns
    -------
    A M x N' matrix containing the particles within the given range. N' is the
    number of particles after filtering.
    """
    elements_to_keep = np.ones_like(beam_matrix[0])
    for i, arr in enumerate(beam_matrix):
        if min_range[i] is not None:
            elements_to_keep = np.where(
                arr < min_range[i], 0, elements_to_keep)
        if max_range[i] is not None:
            elements_to_keep = np.where(
                arr > max_range[i], 0, elements_to_keep)
    elements_to_keep = np.array(elements_to_keep, dtype=bool)
    return beam_matrix[:, elements_to_keep]


def filter_beam_sigma(beam_matrix, max_sigma, w=None):
    """Filter the beam particles keeping only those within a certain number of
    sigmas.

    Parameters
    ----------
    beam_matrix : array
        M x N matrix containing all M components of the N particles.

    max_sigma : array
        Array of size M with the maximum number of sigmas of each component.
        For each compoonent x, particles with x < x_avg - x_std * max_sigma or
        x > x_avg + x_std * max_sigma will be discarded. If max_sigma is None
        for some component, no filtering is performed.

    w : array
        Statistical weight of the particles.

    Returns
    -------
    A M x N' matrix containing the particles within the given range. N' is the
    number of particles after filtering.
    """
    elements_to_keep = np.ones_like(beam_matrix[0])
    for i, arr in enumerate(beam_matrix):
        if max_sigma[i] is not None:
            arr_mean = np.average(arr, weights=w)
            arr_std = weighted_std(arr, weights=w)
            min_range = arr_mean - max_sigma[i] * arr_std
            max_range = arr_mean + max_sigma[i] * arr_std
            elements_to_keep = np.where(
                (arr < min_range) | (arr > max_range), 0, elements_to_keep)
    elements_to_keep = np.array(elements_to_keep, dtype=bool)
    return beam_matrix[:, elements_to_keep]
