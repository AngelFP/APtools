"""This module contains methods for raeding beam data from different particle
tracking codes"""

def read_csrtrack_data_fmt1(file_path):
    """Reads particle data from CSRtrack in fmt1 format and returns it in the
    unis used by APtools.

    Parameters:
    -----------
    file_path : str
        Path to the file with particle data

    Returns:
    --------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.loadtxt(file_path)
    z = data[1:,0]
    x = data[1:,1]
    y = data[1:,2]
    pz = data[1:,3] / (ct.m_e*ct.c**2/ct.e)
    px = data[1:,4] / (ct.m_e*ct.c**2/ct.e)
    py = data[1:,5] / (ct.m_e*ct.c**2/ct.e)
    q = data[1:,6]
    x[1:] += x[0]
    y[1:] += y[0]
    z[1:] += z[0]
    px[1:] += px[0]
    py[1:] += py[0]
    pz[1:] += pz[0]
    return x, y, z, px, py, pz, q
