"""This module defines methods for filtering and selecting beam data"""

import numpy as np


def filter_beam(beam_matrix, min_range, max_range):
    """Filter the beam particles keeping only those within a given range.

    Parameters:
    -----------
    beam_matrix : array
        M x N matrix containing all M components of the N particles
    min_range : array
        Array of size M with the minimum value for each given component. For
        values which are 'None' no filtering is performed.
    max_range : array
        Array of size M with the maximum value for each given component. For
        values which are 'None' no filtering is performed.

    Returns:
    --------
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
