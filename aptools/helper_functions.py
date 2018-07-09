"""This module contains methods needed by other modules"""

import numpy as np

def weighted_std(values, weights=1):
    """Calculates the weighted standard deviation of the given values

    Parameters:
    -----------
    values: array
        Contains the values to be analyzed
    weights : array
        Contains the weights of the values to analyze

    Returns:
    --------
    A float with the value of the standard deviation
    """
    mean_val = np.average(values, weights=np.abs(weights))
    std = np.sqrt(np.average((values-mean_val)**2, weights=np.abs(weights)))
    return std