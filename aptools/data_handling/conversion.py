"""This module contains methods for converting beam data between different
particle tracking and PIC codes"""

from aptools.data_handling.reading import read_beam
from aptools.data_handling.saving import save_beam


def convert_beam(orig_code, final_code, orig_path, final_path, final_file_name,
                 reposition=False, avg_pos=[None, None, None],
                 avg_mom=[None, None, None], n_part=None, read_kwargs={},
                 save_kwargs={}):
    """Converts particle data from one code to another.

    Parameters
    ----------
    orig_code : str
        Name of the tracking or PIC code of the original data. Possible values
        are 'csrtrack', 'astra' and 'openpmd'

    final_code : str
        Name of the tracking or PIC code in which to convert the data. Possible
        values are 'csrtrack', 'astra', 'fbpic' and 'openpmd'

    orig_path : str
        Path of the file containing the original data

    final_path : str
        Path to the folder in which to save the converted data

    final_file_name : str
        Name of the file to save, without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.

    Other Parameters
    ----------------
    **kwargs
        This method takes additional keyword parameters that might be needed
        for some data readers. Currenlty, the only parameter is 'species_name',
        for reading data from PIC codes.
    """
    x, y, z, px, py, pz, q = read_beam(orig_code, orig_path, **read_kwargs)
    beam_data = [x, y, z, px, py, pz, q]
    save_beam(final_code, beam_data, final_path, final_file_name, reposition,
              avg_pos, avg_mom, n_part, **save_kwargs)
