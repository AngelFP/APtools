""" Defines utilities for the data handling modules """
import openpmd_api as io


def get_available_species(file_path):
    """Get a list of available species in an openPMD file.

    Parameters
    ----------
    file_path : str
        Path of the openPMD file.

    Returns
    -------
    list
        A list of strings with the names of the available species.
    """
    series = io.Series(file_path, io.Access.read_only)
    i = list(series.iterations)[0]
    iteration = series.iterations[i]

    return list(iteration.particles)
