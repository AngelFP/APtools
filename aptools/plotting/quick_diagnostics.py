"""This module contains methods for performing and visualizing common beam
diagnostics"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.constants as ct

import aptools.data_analysis.beam_diagnostics as bd
from aptools.data_handling.reading import read_beam
from aptools.plotting.plot_types import scatter_histogram
from aptools.helper_functions import get_only_statistically_relevant_slices


aptools_rc_params = {'axes.linewidth': 0.5,
                     'axes.labelsize': 8,
                     'xtick.major.size': 2,
                     'ytick.major.size': 2,
                     'xtick.major.width': 0.5,
                     'ytick.major.width': 0.5,
                     'xtick.labelsize': 8,
                     'ytick.labelsize': 8,
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'legend.borderaxespad': 1}


def phase_space_overview_from_file(code_name, file_path, **kwargs):
    x, y, z, px, py, pz, q = read_beam(code_name, file_path, **kwargs)
    phase_space_overview(x, y, z, px, py, pz, q)


def phase_space_overview(x, y, z, px, py, pz, q, show=True):
    em_x = bd.normalized_transverse_rms_emittance(x, px, w=q) * 1e6
    em_y = bd.normalized_transverse_rms_emittance(y, py, w=q) * 1e6
    a_x, b_x, g_x = bd.twiss_parameters(x, px, pz, w=q)
    a_y, b_y, g_y = bd.twiss_parameters(y, py, pz, w=q)
    s_x = bd.rms_size(x, w=q)
    s_y = bd.rms_size(y, w=q)
    em_l = bd.longitudinal_rms_emittance(z, px, py, pz, w=q) * 1e6
    dz = z - np.average(z, weights=q)
    s_z = bd.rms_length(z, w=q)
    s_g = bd.relative_rms_energy_spread(pz, py, pz, w=q)
    s_g_sl, w_sl, sl_ed, s_g_sl_av = bd.relative_rms_slice_energy_spread(
        z, px, py, pz, w=q, n_slices=10)
    c_prof, _ = bd.current_profile(z, q, n_slices=50)
    c_peak = max(abs(c_prof))/1e3  # kA
    # s_g_sl_c = s_g_sl[int(len(s_g_sl)/2)]

    # make plot
    plt.figure(figsize=(8, 3))
    with plt.rc_context(aptools_rc_params):
        # x - px
        ax_1 = plt.subplot(131)
        scatter_histogram(x*1e6, px)
        plt.xlabel("x [$\\mu m$]")
        plt.ylabel("$p_x \\ \\mathrm{[m_ec^2/e]}$")
        plt.text(0.1, 0.9, '$\\epsilon_{n,x} = $'
                 + '{}'.format(np.around(em_x, 3))
                 + '$\\ \\mathrm{\\pi \\ \\mu m \\ rad}$',
                 transform=ax_1.transAxes, fontsize=8)
        plt.text(0.1, 0.8, '$\\beta_{x} = $' + '{}'.format(np.around(b_x, 3))
                 + 'm', transform=ax_1.transAxes, fontsize=8)
        plt.text(0.1, 0.7, '$\\alpha_{x} = $' + '{}'.format(np.around(a_x, 3)),
                 transform=ax_1.transAxes, fontsize=8)
        plt.text(0.1, 0.6, '$\\sigma_{x} = $'
                 + '{}'.format(np.around(s_x*1e6, 3))
                 + '$\\ \\mathrm{\\mu m}$', transform=ax_1.transAxes,
                 fontsize=8)
        # y - py
        ax_2 = plt.subplot(132)
        scatter_histogram(y * 1e6, py)
        plt.xlabel("y [$\\mu m$]")
        plt.ylabel("$p_y \\ \\mathrm{[m_ec^2/e]}$")
        plt.text(0.1, 0.9, '$\\epsilon_{n,y} = $'
                 + '{}'.format(np.around(em_y, 3))
                 + '$\\ \\mathrm{\\pi \\ \\mu m \\ rad}$',
                 transform=ax_2.transAxes, fontsize=8)
        plt.text(0.1, 0.8, '$\\beta_{y} = $' + '{}'.format(np.around(b_y, 3))
                 + 'm', transform=ax_2.transAxes, fontsize=8)
        plt.text(0.1, 0.7, '$\\alpha_{y} = $' + '{}'.format(np.around(a_y, 3)),
                 transform=ax_2.transAxes, fontsize=8)
        plt.text(0.1, 0.6, '$\\sigma_{y} = $'
                 + '{}'.format(np.around(s_y*1e6, 3))
                 + '$\\ \\mathrm{\\mu m}$', transform=ax_2.transAxes,
                 fontsize=8)
        # z - pz
        ax_3 = plt.subplot(133)
        scatter_histogram(dz / ct.c * 1e15, pz)
        plt.xlabel("$\\Delta z$ [fs]")
        plt.ylabel("$p_z \\ \\mathrm{[m_ec^2/e]}$")
        plt.text(0.1, 0.9, '$\\epsilon_{L} = $'
                 + '{}'.format(np.around(em_l, 3))
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
        plt.text(0.1, 0.5, '$I_{peak}=$'
                 + '{}'.format(np.around(c_peak, 2)) + ' kA',
                 transform=ax_3.transAxes, fontsize=8)
        plt.tight_layout()
    if show:
        plt.show()


def slice_analysis(x, y, z, px, py, pz, q, n_slices=50, len_slice=None,
                   ene_bins=50, left=0.125, right=0.875, top=0.98, bottom=0.13,
                   xlim=None, ylim=None, add_labels=False, include_twiss=False,
                   fig=None, rasterized_scatter=None, show=True):
    # analyze beam
    current_prof, z_edges = bd.current_profile(z, q, n_slices=n_slices,
                                               len_slice=len_slice)
    ene_spectrum, ene_spec_edgs = bd.energy_spectrum(px, py, pz, w=q,
                                                     bins=ene_bins)
    slice_ene, *_ = bd.energy_profile(
        z, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    slice_ene_sp, *_ = bd.relative_rms_slice_energy_spread(
        z, px, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    sl_tw, sl_w, *_ = bd.slice_twiss_parameters(
        z, x, px, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    alpha_x, *_ = get_only_statistically_relevant_slices(
        sl_tw[0], sl_w, replace_with_nans=True)
    beta_x, *_ = get_only_statistically_relevant_slices(
        sl_tw[1], sl_w, replace_with_nans=True)
    sl_tw, *_ = bd.slice_twiss_parameters(
        z, y, py, pz, w=q, n_slices=n_slices, len_slice=len_slice)
    alpha_y, *_ = get_only_statistically_relevant_slices(
        sl_tw[0], sl_w, replace_with_nans=True)
    beta_y, *_ = get_only_statistically_relevant_slices(
        sl_tw[1], sl_w, replace_with_nans=True)
    slice_em_x, *_ = bd.normalized_transverse_rms_slice_emittance(
        z, x, px, w=q, n_slices=n_slices, len_slice=len_slice)
    slice_em_y, *_ = bd.normalized_transverse_rms_slice_emittance(
        z, y, py, w=q, n_slices=n_slices, len_slice=len_slice)
    s_z = bd.rms_length(z, w=q)
    len_fwhm = bd.fwhm_length(z, q, n_slices=n_slices, len_slice=len_slice)
    ene_sp_tot = bd.relative_rms_energy_spread(px, py, pz, w=q)

    # perform operations
    gamma = np.sqrt(1 + px**2 + py**2 + pz**2)
    ene = gamma * ct.m_e*ct.c**2/ct.e * 1e-9  # GeV
    z_center = np.average(z, weights=q)
    dz = z_edges[1] - z_edges[0]
    slice_z = (z_edges[1:] - dz/2 - z_center) * 1e6  # micron
    current_prof = np.abs(current_prof) * 1e-3  # kA
    peak_current = np.nanmax(current_prof)
    s_t = s_z * 1e15/ct.c
    len_fwhm *= 1e15/ct.c  # fs
    slice_ene *= ct.m_e*ct.c**2/ct.e * 1e-9  # GeV
    ene_spec_edgs = ene_spec_edgs[:-1] + (ene_spec_edgs[1]-ene_spec_edgs[0])/2
    ene_spec_edgs *= ct.m_e*ct.c**2/ct.e * 1e-9  # GeV
    slice_ene_sp *= 1e2  # %
    ene_sp_tot *= 1e2  # %
    slice_em_x *= 1e6  # micron
    slice_em_y *= 1e6  # micron
    max_beta = np.nanmax(beta_x)
    if max_beta <= 0.1:
        beta_units = 'mm'
        beta_x *= 1e3
        beta_y *= 1e3
    else:
        beta_units = 'm'
    max_ene = np.nanmax(ene)
    if max_ene <= 1:
        ene_units = 'MeV'
        ene *= 1e3
        ene_spec_edgs *= 1e3
    else:
        ene_units = 'GeV'
    ene_mean = np.average(ene, weights=q)

    # make plot
    if include_twiss:
        nrows = 3
        hr = [2.5, 1, 1]
        fh = 3.3
    else:
        nrows = 2
        hr = [2.5, 1]
        fh = 2.5
    if fig is None:
        fig = plt.figure(figsize=(4, fh))
    gs = gridspec.GridSpec(nrows, 2, height_ratios=hr,
                           width_ratios=[1, 0.02], hspace=0.1, wspace=0.05,
                           figure=fig, left=left, right=right,
                           top=top, bottom=bottom)
    leg_frac = 0.25  # space to reserve for legend

    with plt.rc_context(aptools_rc_params):
        ax_or = plt.subplot(gs[0])
        pscatt = scatter_histogram((z-z_center)*1e6, ene, bins=300,
                                   weights=np.abs(q)*1e15,
                                   rasterized=rasterized_scatter)
        plt.ylabel('Energy [{}]'.format(ene_units))
        plt.tick_params(axis='x', which='both', labelbottom=False)
        params_text = ('$\\langle E \\rangle = '
                       + '{:0.1f}$ {}\n'.format(ene_mean, ene_units)
                       + '$\\sigma_\\mathrm{E,rel}='
                       + '{:0.1f}$ %\n'.format(ene_sp_tot)
                       + '$I_\\mathrm{peak}='
                       + '{:0.1f}$ kA\n'.format(peak_current)
                       + '$\\sigma_t='
                       + '{:0.1f}$ fs'.format(s_t))
        plt.text(0.98, 0.95, params_text, transform=ax_or.transAxes,
                 fontsize=6, horizontalalignment='right',
                 verticalalignment='top')
        if add_labels:
            plt.text(0.03, 0.05, '(a)', transform=ax_or.transAxes, fontsize=6,
                     horizontalalignment='left', verticalalignment='bottom')

        if xlim is None:
            xlim = list(plt.xlim())
            xlim[0] -= (xlim[1] - xlim[0])/8
            xlim[1] += (xlim[1] - xlim[0])/3
        plt.xlim(xlim)

        if ylim is None:
            ylim = list(plt.ylim())
            ylim[0] -= (ylim[1] - ylim[0])/3
        plt.ylim(ylim)

        # current profile plot
        z_or = ax_or.get_zorder()
        pos = list(ax_or.get_position().bounds)
        pos[3] /= 5
        ax_or.patch.set_alpha(0)
        ax = fig.add_axes(pos)
        ax.set_zorder(z_or-1)
        plt.plot(slice_z, current_prof, c='k', lw=0.5, alpha=0.5)
        plt.fill_between(slice_z, current_prof, facecolor='tab:gray',
                         alpha=0.3)
        ax.spines['left'].set_position('zero')
        ax.spines['left'].set_color('tab:grey')
        ax.tick_params(axis='y', colors='tab:grey', labelsize=6,
                       direction="in", pad=-4)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plt.tick_params(axis='x', which='both', labelbottom=False)
        for label in ax.yaxis.get_ticklabels():
            label.set_horizontalalignment('left')
            label.set_verticalalignment('bottom')
        plt.xlim(xlim)
        ylim_c = list(plt.ylim())
        ylim_c[0] = 0
        plt.ylim(ylim_c)
        plt.ylabel('I [kA]', color='tab:gray', fontsize=6)

        # energy profile plot
        pos = list(ax_or.get_position().bounds)
        pos[2] /= 8
        ax = fig.add_axes(pos)
        ax.set_zorder(z_or-1)
        plt.plot(ene_spectrum, ene_spec_edgs, c='k', lw=0.5, alpha=0.5)
        plt.fill_betweenx(ene_spec_edgs, ene_spectrum, facecolor='tab:gray',
                          alpha=0.3)
        plt.gca().axis('off')
        plt.ylim(ylim)
        xlim_e = list(plt.xlim())
        xlim_e[0] = 0
        plt.xlim(xlim_e)

        # colorbar
        ax = plt.subplot(gs[1])
        matplotlib.colorbar.Colorbar(ax, pscatt, label='Q [fC]')

        # slice parameters plot
        plt.subplot(gs[2])
        l1 = plt.plot(slice_z, slice_ene_sp, lw=1, c='tab:green',
                      label='$\\sigma_\\gamma/\\gamma$')
        plt.ylabel('$\\sigma_\\gamma/\\gamma$ [%]')
        if include_twiss:
            plt.tick_params(axis='x', which='both', labelbottom=False)
        else:
            plt.xlabel('$\\Delta z \\ [\\mathrm{\\mu m}]$')
        # make room for legend
        # ylim = list(plt.ylim())
        # ylim[1] += (ylim[1] - ylim[0]) * leg_frac
        plt.xlim(xlim)
        # plt.ylim(ylim)

        ax = plt.twinx()
        l2 = plt.plot(slice_z, slice_em_x, lw=1, c='tab:blue',
                      label='$\\epsilon_{n,x}$')
        l3 = plt.plot(slice_z, slice_em_y, lw=1, c='tab:orange',
                      label='$\\epsilon_{n,y}$')
        plt.ylabel('$\\epsilon_{n} \\ [\\mathrm{\\mu m}]$')
        # make room for legend
        # ylim = list(plt.ylim())
        # ylim[1] += (ylim[1] - ylim[0]) * leg_frac
        # plt.ylim(ylim)
        lines = l1 + l2 + l3
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels, fontsize=6, frameon=False,
                   loc='center right', borderaxespad=0.3)
        if add_labels:
            plt.text(0.03, 0.05, '(b)', transform=plt.gca().transAxes,
                     fontsize=6, horizontalalignment='left',
                     verticalalignment='bottom')

        if include_twiss:
            plt.subplot(gs[4])
            l1 = plt.plot(slice_z, beta_x, lw=1, c='tab:blue',
                          label='$\\beta_x$')
            l2 = plt.plot(slice_z, beta_y, lw=1, c='tab:orange',
                          label='$\\beta_y$')
            plt.xlabel('$\\Delta z \\ [\\mathrm{\\mu m}]$')
            plt.ylabel('$\\beta$ [{}]'.format(beta_units))
            # make room for legend
            ylim = list(plt.ylim())
            ylim[1] += (ylim[1] - ylim[0]) * leg_frac
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.twinx()
            l3 = plt.plot(slice_z, alpha_x, lw=1, c='tab:blue', ls='--',
                          label='$\\alpha_x$')
            l4 = plt.plot(slice_z, alpha_y, lw=1, c='tab:orange', ls='--',
                          label='$\\alpha_y$')
            lines = l1 + l2 + l3 + l4
            labels = [line.get_label() for line in lines]
            # make room for legend
            # ylim = list(plt.ylim())
            # ylim[1] += (ylim[1] - ylim[0]) * leg_frac
            # plt.ylim(ylim)
            plt.legend(lines, labels, fontsize=6, ncol=1, frameon=False,
                       loc='center right', borderaxespad=0.3,
                       labelspacing=0.20)
            if add_labels:
                plt.text(0.03, 0.05, '(c)', transform=plt.gca().transAxes,
                         fontsize=6, horizontalalignment='left',
                         verticalalignment='bottom')
            plt.ylabel('$\\alpha$')
    if show:
        plt.show()


def lon_phase_space(
        x, y, z, px, py, pz, q, n_slices=50, len_slice=None, ene_bins=50,
        xlim=None, ylim=None, show_text=True, x_proj=True, y_proj=True,
        cbar=True, left=0.125, right=0.875, top=0.98, bottom=0.13, fig=None,
        rasterized_scatter=None, show=True):
    # analyze beam
    current_prof, z_edges = bd.current_profile(z, q, n_slices=n_slices,
                                               len_slice=len_slice)
    ene_spectrum, ene_spec_edgs = bd.energy_spectrum(px, py, pz, w=q,
                                                     bins=ene_bins)
    s_z = bd.rms_length(z, w=q)
    len_fwhm = bd.fwhm_length(z, q, n_slices=n_slices, len_slice=len_slice)
    ene_sp_tot = bd.relative_rms_energy_spread(px, py, pz, w=q)

    # perform operations
    gamma = np.sqrt(1 + px**2 + py**2 + pz**2)
    ene = gamma * ct.m_e*ct.c**2/ct.e * 1e-9  # GeV
    z_center = np.average(z, weights=q)
    dz = z_edges[1] - z_edges[0]
    slice_z = (z_edges[1:] - dz/2 - z_center) * 1e6  # micron
    current_prof = np.abs(current_prof) * 1e-3  # kA
    peak_current = np.nanmax(current_prof)
    s_t = s_z * 1e15/ct.c
    len_fwhm *= 1e15/ct.c  # fs
    ene_spec_edgs = ene_spec_edgs[:-1] + (ene_spec_edgs[1]-ene_spec_edgs[0])/2
    ene_spec_edgs *= ct.m_e*ct.c**2/ct.e * 1e-9  # GeV
    ene_sp_tot *= 1e2  # %
    max_ene = np.nanmax(ene)
    if max_ene <= 1:
        ene_units = 'MeV'
        ene *= 1e3
        ene_spec_edgs *= 1e3
    else:
        ene_units = 'GeV'
    ene_mean = np.average(ene, weights=q)

    # make plot
    if fig is None:
        fig = plt.figure(figsize=(4, 2.5))
    if cbar:
        gs = gridspec.GridSpec(
            1, 2, width_ratios=[1, 0.02], hspace=0.1, wspace=0.05,
            figure=fig, left=left, right=right, top=top, bottom=bottom)
    else:
        gs = gridspec.GridSpec(
            1, 1, figure=fig, left=left, right=right, top=top, bottom=bottom)
    with plt.rc_context(aptools_rc_params):
        ax_or = plt.subplot(gs[0])
        pscatt = scatter_histogram((z-z_center)*1e6, ene, bins=300,
                                   weights=np.abs(q)*1e15,
                                   rasterized=rasterized_scatter)
        plt.xlabel('$\\Delta z \\ [\\mathrm{\\mu m}]$')
        plt.ylabel('Energy [{}]'.format(ene_units))
        if show_text:
            params_text = ('$\\langle E \\rangle = '
                           + '{:0.1f}$ {}\n'.format(ene_mean, ene_units)
                           + '$\\sigma_\\mathrm{E,rel}='
                           + '{:0.1f}$ %\n'.format(ene_sp_tot)
                           + '$I_\\mathrm{peak}='
                           + '{:0.1f}$ kA\n'.format(peak_current)
                           + '$\\sigma_t='
                           + '{:0.1f}$ fs'.format(s_t))
            plt.text(0.98, 0.95, params_text, transform=ax_or.transAxes,
                     fontsize=6, horizontalalignment='right',
                     verticalalignment='top')

        if xlim is not None:
            plt.xlim(xlim)
        else:
            xlim = list(plt.xlim())
            if y_proj:
                xlim[0] -= (xlim[1] - xlim[0])/8
            if show_text:
                xlim[1] += (xlim[1] - xlim[0])/3
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        else:
            ylim = list(plt.ylim())
            if x_proj:
                ylim[0] -= (ylim[1] - ylim[0])/3
            plt.ylim(ylim)

        # current profile plot
        if x_proj:
            z_or = ax_or.get_zorder()
            pos = list(ax_or.get_position().bounds)
            pos[3] /= 5
            ax_or.patch.set_alpha(0)
            ax = fig.add_axes(pos)
            ax.set_zorder(z_or-1)
            plt.plot(slice_z, current_prof, c='k', lw=0.5, alpha=0.5)
            plt.fill_between(
                slice_z, current_prof, facecolor='tab:gray', alpha=0.3)
            ax.spines['left'].set_position('zero')
            ax.spines['left'].set_color('tab:grey')
            ax.tick_params(
                axis='y', colors='tab:grey', labelsize=6, direction="in",
                pad=-4)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            plt.tick_params(axis='x', which='both', labelbottom=False)
            for label in ax.yaxis.get_ticklabels():
                label.set_horizontalalignment('left')
                label.set_verticalalignment('bottom')
            plt.xlim(xlim)
            ylim_c = list(plt.ylim())
            ylim_c[0] = 0
            plt.ylim(ylim_c)
            plt.ylabel('I [kA]', color='tab:gray', fontsize=6)

        # energy profile plot
        if y_proj:
            z_or = ax_or.get_zorder()
            pos = list(ax_or.get_position().bounds)
            pos[2] /= 8
            ax_or.patch.set_alpha(0)
            ax = fig.add_axes(pos)
            ax.set_zorder(z_or-1)
            plt.plot(ene_spectrum, ene_spec_edgs, c='k', lw=0.5, alpha=0.5)
            plt.fill_betweenx(
                ene_spec_edgs, ene_spectrum, facecolor='tab:gray', alpha=0.3)
            plt.gca().axis('off')
            plt.ylim(ylim)
            xlim_e = list(plt.xlim())
            xlim_e[0] = 0
            plt.xlim(xlim_e)

        # colorbar
        if cbar:
            ax = plt.subplot(gs[1])
            matplotlib.colorbar.Colorbar(ax, pscatt, label='Q [fC]')

    if show:
        plt.show()
