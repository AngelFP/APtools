import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import ticker
from matplotlib.colorbar import Colorbar
import matplotlib.patheffects as path_effects
from matplotlib import colors


def add_projection(
        x, bins, main_ax, subplot_spec, fig, orientation='horizontal'):
    x_proj, x_bins = np.histogram(x, bins=bins)
    x_pos = x_bins[1:] - (x_bins[1]-x_bins[0])

    if orientation == 'horizontal':
        gs_p = gs.GridSpecFromSubplotSpec(
            2, 1, subplot_spec, height_ratios=[1, 0.2])
        ax_p = fig.add_subplot(gs_p[-1])
    elif orientation == 'vertical':
        gs_p = gs.GridSpecFromSubplotSpec(
            1, 2, subplot_spec, width_ratios=[0.2, 1])
        ax_p = fig.add_subplot(gs_p[0])

    ax_p.patch.set_alpha(0)

    if orientation == 'horizontal':
        ax_p.plot(x_pos, x_proj, c='k', lw=0.5, alpha=0.5)
        ax_p.fill_between(x_pos, x_proj, facecolor='tab:gray', alpha=0.3)
        xlim = main_ax.get_xlim()
        ax_p.set_xlim(xlim)
        ylim = list(ax_p.get_ylim())
        ylim[0] = 0
        ax_p.set_ylim(ylim)
    elif orientation == 'vertical':
        ax_p.plot(x_proj, x_pos, c='k', lw=0.5, alpha=0.5)
        ax_p.fill_betweenx(x_pos, x_proj, facecolor='tab:gray', alpha=0.3)
        ylim = main_ax.get_ylim()
        ax_p.set_ylim(ylim)
        xlim = list(ax_p.get_xlim())
        xlim[0] = 0
        ax_p.set_xlim(xlim)
    ax_p.axis('off')


def create_vertical_colorbars(
        images, labels, subplot_spec, fig=None, n_ticks=3, **kwargs):
    if not isinstance(images, list):
        images = [images]
    if not isinstance(labels, list):
        labels = [labels]
    n_cbars = len(images)
    cbar_gs = gs.GridSpecFromSubplotSpec(
        n_cbars, 1, subplot_spec=subplot_spec, **kwargs)
    if fig is None:
        fig = plt.gcf()
    for image, label, cbar_ss in zip(images, labels, cbar_gs):
        ax = fig.add_subplot(cbar_ss)
        tick_locator = ticker.MaxNLocator(nbins=n_ticks)
        Colorbar(ax, image, ticks=tick_locator, label=label)


def add_text(ax, x, y, text, **kwargs):
    fc = colors.to_rgba('white')
    # fc[:-1] + (0.7,)
    ec = colors.to_rgba('tab:gray')
    bbox = dict(
        boxstyle="round",
        ec=ec,
        fc=fc,
        alpha=0.7
        )
    label = ax.text(
        x, y, text, transform=ax.transAxes, fontsize=8, bbox=bbox, **kwargs)
    label.set_path_effects(
        [path_effects.Stroke(linewidth=1, foreground='white'),
         path_effects.Normal()])
