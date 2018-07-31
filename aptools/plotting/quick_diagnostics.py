"""This module contains methods for performing and visualizing common beam
diagnostics"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as ct

import aptools.beam_diagnostics as bd


def phase_space_overview(x, y, z, px, py, pz, q):
    em_x = bd.normalized_transverse_rms_emittance(x, px, q) * 1e6
    em_y = bd.normalized_transverse_rms_emittance(y, py, q) * 1e6
    a_x, b_x = bd.twiss_parameters(x, px, pz, q)
    a_y, b_y = bd.twiss_parameters(y, py, pz, q)
    s_x = bd.rms_size(x, q)
    s_y = bd.rms_size(y, q)
    em_l = bd.longitudinal_rms_emittance(z, px, py, pz, w=q) * 1e6
    dz = z - np.average(z, weights=q)
    s_z = bd.rms_length(z, q)
    s_g = bd.relative_rms_energy_spread(pz, py, pz, q)
    s_g_sl, w_sl, sl_ed = bd.relative_rms_slice_energy_spread(z, px, py, pz, q,
                                                              10)
    s_g_sl_av = np.average(s_g_sl, weights=w_sl)
    #s_g_sl_c = s_g_sl[int(len(s_g_sl)/2)]
    plt.figure(figsize=(8,3))
    # x - px
    ax_1 = plt.subplot(131)
    plt.plot(x * 1e6, px, '.', ms=1)
    plt.xlabel("x [$\\mu m$]")
    plt.ylabel("$p_x \\ \\mathrm{[m_ec^2/e]}$")
    plt.text(0.1, 0.9, '$\\epsilon_{n,x} = $' + '{}'.format(np.around(em_x, 3))
             + '$\\ \\mathrm{\\pi \\ \\mu m \\ rad}$',
             transform=ax_1.transAxes, fontsize=8)
    plt.text(0.1, 0.8, '$\\beta_{x} = $' + '{}'.format(np.around(b_x, 3))
             + 'm', transform=ax_1.transAxes, fontsize=8)
    plt.text(0.1, 0.7, '$\\alpha_{x} = $' + '{}'.format(np.around(a_x, 3)),
             transform=ax_1.transAxes, fontsize=8)
    plt.text(0.1, 0.6, '$s_{x} = $' + '{}'.format(np.around(s_x*1e6, 3))
             + '$\\ \\mathrm{\\mu m}$', transform=ax_1.transAxes, fontsize=8)
    # y - py
    ax_2 = plt.subplot(132)
    plt.plot(y * 1e6, py, '.', ms=1)    
    plt.xlabel("y [$\\mu m$]")
    plt.ylabel("$p_y \\ \\mathrm{[m_ec^2/e]}$")
    plt.text(0.1, 0.9, '$\\epsilon_{n,y} = $' + '{}'.format(np.around(em_y, 3))
             + '$\\ \\mathrm{\\pi \\ \\mu m \\ rad}$',
             transform=ax_2.transAxes, fontsize=8)
    plt.text(0.1, 0.8, '$\\beta_{y} = $' + '{}'.format(np.around(b_y, 3))
             + 'm', transform=ax_2.transAxes, fontsize=8)
    plt.text(0.1, 0.7, '$\\alpha_{y} = $' + '{}'.format(np.around(a_y, 3)),
             transform=ax_2.transAxes, fontsize=8)
    plt.text(0.1, 0.6, '$s_{y} = $' + '{}'.format(np.around(s_y*1e6, 3))
             + '$\\ \\mathrm{\\mu m}$', transform=ax_2.transAxes, fontsize=8)
    # z - pz
    ax_3 = plt.subplot(133)
    plt.plot(dz / ct.c * 1e15, pz, '.', ms=1)
    plt.xlabel("$\\Delta z$ [fs]")
    plt.ylabel("$p_z \\ \\mathrm{[m_ec^2/e]}$")
    plt.text(0.1, 0.9, '$\\epsilon_{L} = $' + '{}'.format(np.around(em_l, 3))
             + '$\\ \\mathrm{\\pi \\ \\mu m}$', transform=ax_3.transAxes,
             fontsize=8)
    plt.text(0.1, 0.8, '$\\sigma_\\gamma/\\gamma=$'
             + '{}'.format(np.around(s_g*1e2, 3)) + '$\\%$',
             transform=ax_3.transAxes, fontsize=8)
    plt.text(0.1, 0.7, '$\\sigma^s_\\gamma/\\gamma=$'
             + '{}'.format(np.around(s_g_sl_av*1e2, 3)) + '$\\%$',
             transform=ax_3.transAxes, fontsize=8)
    plt.text(0.1, 0.6, '$\\sigma_z=$'
             + '{}'.format(np.around(s_z/ct.c*1e15, 3)) + ' fs',
             transform=ax_3.transAxes, fontsize=8)
    plt.tight_layout()
