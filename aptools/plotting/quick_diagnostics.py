"""This module contains methods for performing and visualizing common beam
diagnostics"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct

import aptools.beam_diagnostics as bd


def phase_space_overview(x, y, z, px, py, pz, q):
    em_x = bd.normalized_transverse_rms_emittance(x, px, q) * 1e6
    em_y = bd.normalized_transverse_rms_emittance(y, py, q) * 1e6
    dz = z - np.average(z, weights=q)
    plt.figure(figsize=(8,3))
    ax_1 = plt.subplot(131)
    plt.plot(x * 1e6, px, '.', ms=1)
    plt.xlabel("x [$\\mu m$]")
    plt.ylabel("$p_x \\ \\mathrm{[m_ec^2/e]}$")
    plt.text(0.3, 0.9, '$\\epsilon_{n,x} = $' + '{}'.format(np.around(em_x, 3)) + '$\\ \\mathrm{\\mu m \\ rad}$', transform=ax_1.transAxes, fontsize=8)
    ax_2 = plt.subplot(132)
    plt.plot(y * 1e6, py, '.', ms=1)    
    plt.xlabel("y [$\\mu m$]")
    plt.ylabel("$p_y \\ \\mathrm{[m_ec^2/e]}$")
    plt.text(0.3, 0.9, '$\\epsilon_{n,y} = $' + '{}'.format(np.around(em_y, 3)) + '$\\ \\mathrm{\\mu m \\ rad}$', transform=ax_2.transAxes, fontsize=8)
    ax_3 = plt.subplot(133)
    plt.plot(dz / ct.c * 1e15, pz, '.', ms=1)
    plt.xlabel("$\\Delta z$ [fs]")
    plt.ylabel("$p_z \\ \\mathrm{[m_ec^2/e]}$")
    plt.tight_layout()
