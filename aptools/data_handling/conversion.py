"""This module contains methods for converting beam data between different
particle tracking and PIC codes"""

from aptools.data_handling.reading import read_beam
from aptools.data_handling.saving import save_beam


def convert_beam(orig_code, final_code, orig_path, final_path, final_file_name,
                 reposition=False, avg_pos=[None, None, None], n_part=None,
                 species_name=None):
    """Converts particle data from one code to another.

    Parameters:
    -----------
    orig_code : str
        Name of the tracking or PIC code of the original data. Possible values
        are 'csrtrack', 'astra' and 'openpmd'

    final_code : str
        Name of the tracking or PIC code in which to convert the data. Possible
        values are 'csrtrack', 'astra' and 'fbpic'

    orig_path : str
        Path of the file containing the original data

    final_path : str
        Path to the folder in which to save the converted data

    final_file_name : str
        Name of the file to save, without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        centered in the coordinates specified in avg_pos

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.

    species_name : std
        Only required for reading data from PIC codes. Name of the particle
        species.
    """
    x, y, z, px, py, pz, q = read_beam(orig_code, orig_path, species_name)
    beam_data = [x, y, z, px, py, pz, q]
    save_beam(final_code, beam_data, final_path, final_file_name, reposition,
                          avg_pos, n_part)
