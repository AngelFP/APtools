""" Contains methods for making different types of plots """

import matplotlib.pyplot as plt
import numpy as np


def scatter_histogram(x, y, bins=[300, 300], range=None, weights=None,
                      cmap='plasma', s=1, edgecolor='none', ax=None, **kwargs):
    """
    Does a scatter plot from the histogram of a particle distribution.
    """
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range,
                                            weights=weights)
    x_grid = xedges[1:] - np.abs(xedges[1]-xedges[0])/2
    y_grid = yedges[1:] - np.abs(yedges[1]-yedges[0])/2
    X, Y = np.meshgrid(x_grid, y_grid)
    counts = counts.T.flatten()
    X = X.flatten()
    Y = Y.flatten()
    # filter out empty areas
    filt = np.where(np.abs(counts) > 0)
    counts = counts[filt]
    # determine order in which the scatter dots will be drawn so that higher
    # values appear on top
    draw_order = np.argsort(np.abs(counts))
    counts = counts[draw_order]
    # normalize counts
    # counts /= np.max(counts)
    # apply filter and draw order to X and Y arrays
    X = X[filt][draw_order]
    Y = Y[filt][draw_order]
    if ax is None:
        sc = plt.scatter(X, Y, c=counts, s=s, cmap=cmap, edgecolor=edgecolor,
                         **kwargs)
        ax = plt.gca()
    else:
        sc = ax.scatter(X, Y, c=counts, s=s, cmap=cmap, edgecolor=edgecolor,
                        **kwargs)
    ax.set_xlim((np.min(xedges), np.max(xedges)))
    ax.set_ylim((np.min(yedges), np.max(yedges)))
    return sc
