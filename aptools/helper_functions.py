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

def join_infile_path(*paths):
    """
    Join path components using '/' as separator.
    This method is defined as an alternative to os.path.join, which uses '\\'
    as separator in Windows environments and is therefore not valid to navigate
    within data files.

    Parameters:
    -----------
    *paths: all strings with path components to join

    Returns:
    --------
    A string with the complete path using '/' as separator.
    """
    # Join path components
    path = '/'.join(paths)
    # Correct double slashes, if any is present
    path = path.replace('//', '/')
    return path