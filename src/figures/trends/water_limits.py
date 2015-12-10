#!/usr/bin/env python2

import os
import seaborn as sns
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
#from matplotlib import style

def main():
    """
    Does a quick re-sampling of flux tower observation form hourly to daily
    timesteps
    """

    month_lab = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pload = lambda x: pickle.load(open(x, 'rb'))

    model_dict = pload(MODFILE)
    obs_data = pload(OBSFILE)

    temp = model_dict['Exp_1']
    #temp = obs_data
    temp['GPP2'] = -temp['GPP']*12
    temp['WUE'] = -temp['GPP']/(temp['Qle']/18/2.45)
    temp['Year'] = temp.index.year
    temp['Month'] = temp.index.month

    sns.set_style("whitegrid")
    #plt.rcParams.update({'mathtext.default': 'regular'})
    kws = dict(s=50, linewidth=.5, edgecolor="w", alpha=0.7)

    wue_plot = sns.FacetGrid(temp, col="Year", hue="Month", col_wrap=4, size=3)
    wue_plot.map(plt.scatter, "IntSWC", "WUE", **kws)
    wue_plot.set_axis_labels('$z^{-1}\int^{z}_{0}\\theta_{s} dz$ (m$^{3}$ m$^{-3}$)',  \
                             'WUE (mol CO$_{2}$ mol$^{-1}$ H$_{2}$O)')

    wue_plot.set(xlim=(0.18, 0.34), ylim=(0, 6), xticks=np.arange(0.2, 0.35, 0.02))

#    for wue0 in wue_plot.get_xticklabels():
#        wue0.set_rotation(45)
    print len(wue_plot)

    leg = plt.legend(loc='right', labels=month_lab, ncol=4, bbox_to_anchor=(2.8, 0.5), \
                     borderpad=2)
    leg.get_frame().set_edgecolor('black')

    wue_plot.fig.subplots_adjust(wspace=.08, hspace=0.15, bottom=0.08)

    plt.savefig(WUEPLOT)
    return 1

    sns.set_style("darkgrid")
    wue_plot = sns.FacetGrid(temp, col="Year", hue="Month", col_wrap=4, size=3)
    wue_plot.map(plt.scatter, "Qle", "GPP2", **kws)
    wue_plot.set_axis_labels('LE (MJ m$^{-2}$ d$^{-1}$)', 'GPP (gC m$^{-2}$ d$^{-1}$)')
    wue_plot.set(xlim=(0, 20), ylim=(0, 10))
    wue_plot.fig.subplots_adjust(wspace=.08)

    plt.show()

    return 1


if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    INFILE = FILEPATH + "daily_inputs.pkl"
    OBSFILE = FILEPATH + "daily_tower_fluxes.pkl"
    MODFILE = FILEPATH + "daily_fluxes.pkl"

    # Figure names
    FIGPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/")
    WUEPLOT = FIGPATH + "wue_swc_rel.pdf"

    main()
