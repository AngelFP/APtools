""" This module contains the definition of custom APtools rc params """

from matplotlib import rcParamsDefault


custom_rc_params = {'axes.linewidth': 0.5,
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


rc_params = {**rcParamsDefault, **custom_rc_params}