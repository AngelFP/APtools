"""This module contains methods needed by other modules"""

import numpy as np

def create_beam_slices(z, n_slices=10, len_slice=None):
    """Calculates the slice limits along z of a partile distribution for a
    given number of slices or slice length.

    Parameters:
    -----------
    z : array
        Contains the longitudinal position of the particles in units of meters
    n_slices : array
        Number of longitudinal slices in which to divite the particle
        distribution. Not used if len_slice is specified.
    len_slice : array
        Length of the longitudinal slices. If not None, replaces n_slices.

    Returns:
    --------
    A tuple containing an array with the slice limits and integer with the
    number of slices, which might have been redefined.
    """
    max_z = np.max(z)
    min_z = np.min(z)
    if len_slice is None:
        slice_lims = np.linspace(min_z, max_z, n_slices+1)
    else:
        slice_lims = np.arange(min_z, max_z, len_slice)
        slice_lims = np.append(slice_lims, max_z)
        n_slices = len(slice_lims)-1
    return slice_lims, n_slices

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

def slope_of_correlation(y, x, w=None):
    """Calculates the slope of the correlation between two variables x and y
    according to y = slope*x

    Parameters:
    -----------
    y: array
        Contains the x values
    y: array
        Contains the y values
    w : array
        Contains the weights of the values

    Returns:
    --------
    A float with the value of the slope
    """
    a = np.average(y*x, weights=w)
    b = np.average(y, weights=w)
    c = np.average(x, weights=w)
    d = np.average(x**2, weights=w)
    return (a-b*c) / (d-c**2)

def remove_correlation(x, y, w=None, order=1):
    """Removes the correlation between two variables x and y, where y=y(x), up
    to the desired order.

    Parameters:
    -----------
    y: array
        Contains the x values
    y: array
        Contains the y values
    w : array
        Contains the weights of the values
    order : int
        Determines the order of the polynomial fit and, thus, the higher
        correlaton order to remove.
    Returns:
    --------
    An array containing the new values of y
    """
    fit_coefs = np.polyfit(x, y, order, w=w)
    for i, coef in enumerate(reversed(fit_coefs[:-1])):
        y = y - coef * x**(i+1)
    return y

def reposition_bunch(beam_data, avg_pos):
    """Reposition bunch with the specified averages"""
    q = beam_data[6]
    for i, new_avg in enumerate(avg_pos):
        if new_avg is not None:
            current_avg = np.average(beam_data[i], weights=q)
            beam_data[i] += new_avg - current_avg

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

def filter_nans(data, data_weights):
    """
    Removes NaN values from a data array and is corresponding value in the
    weights array.

    Parameters:
    -----------
    data: data array to filter
    data_weights: array with the same size as data containing the weights

    Returns:
    --------
    Filtered data and data_weights arrays.
    """
    filter_idx = np.isfinite(data)
    data_weights_f = data_weights[filter_idx]
    data_f = data[filter_idx]
    return data_f, data_weights_f