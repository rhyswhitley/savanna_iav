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

    pload = lambda x: pickle.load(open(x, 'rb'))

    model_dict = pload(MODFILE)
    obs_data = pload(OBSFILE)

    model_out = add_to_data(model_dict['Exp_1'])
    tower_obs = add_to_data(obs_data)

    create_obswater_plots(tower_obs, 'Obs')
    create_modwater_plots(model_out, 'Mod')

def add_to_data(temp_df):
    temp_df['GPP2'] = -temp_df['GPP']*12
    temp_df['WUE'] = -temp_df['GPP']/(temp_df['Qle']/18/2.45)
    temp_df['Year'] = temp_df.index.year
    temp_df['Month'] = temp_df.index.month
    return temp_df

def create_modwater_plots(temp_df, label="Mod"):
    #x_label_custom = '$z^{-1}\int^{z}_{0}\\theta_{s} dz$ (m$^{3}$ m$^{-3}$)'
    x_label_custom = '$\\theta_{s,z=20cm}$ (m$^{3}$ m$^{-3}$)'
    xlab = "SWC20"
    p1 = plot_swcrel(temp_df, xlab, "WUE")
    p1.set_axis_labels(x_label_custom,  \
                             'WUE (mol CO$_{2}$ mol$^{-1}$ H$_{2}$O)')
    plt.savefig(FIGPATH + "{0}_WUExSWC.pdf".format(label))

    p2 = plot_swcrel(temp_df, xlab, "GPP2")
    p2.set_axis_labels(x_label_custom,  \
                             'GPP (gC m$^{-2}$ d$^{-1}$')
    plt.savefig(FIGPATH + "{0}_GPPxSWC.pdf".format(label))

    p3 = plot_swcrel(temp_df, xlab, "Qle")
    p3.set_axis_labels(x_label_custom,  \
                             'LE (MJ m$^{-2}$ d$^{-1}$')
    plt.savefig(FIGPATH + "{0}_QLExSWC.pdf".format(label))

    return None

def create_obswater_plots(temp_df, label="Mod"):
    x_label_custom = '$\\theta_{s,10cm}$ (m$^{3}$ m$^{-3}$)'
    p1 = plot_swcrel(temp_df, "SoilMoist10", "WUE")
    p1.set_axis_labels(x_label_custom,  \
                             'WUE (mol CO$_{2}$ mol$^{-1}$ H$_{2}$O)')
    plt.savefig(FIGPATH + "{0}_WUExSWC.pdf".format(label))

    p2 = plot_swcrel(temp_df, "SoilMoist10", "GPP2")
    p2.set_axis_labels(x_label_custom,  \
                             'GPP (gC m$^{-2}$ d$^{-1}$')
    plt.savefig(FIGPATH + "{0}_GPPxSWC.pdf".format(label))

    p3 = plot_swcrel(temp_df, "SoilMoist10", "Qle")
    p3.set_axis_labels(x_label_custom,  \
                             'LE (MJ m$^{-2}$ d$^{-1}$')
    plt.savefig(FIGPATH + "{0}_QLExSWC.pdf".format(label))

    return None

def plot_swcrel(data, xlabel, ylabel):

    month_lab = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', \
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.set_style("ticks")
    plt.rcParams.update({'mathtext.default': 'regular'})
    kws = dict(s=20, linewidth=.5, edgecolor="none", alpha=0.3)

    wue_plot = sns.FacetGrid(data, hue="Month", size=5)
    wue_plot.map(plt.scatter, xlabel, ylabel, **kws)

    ymax = np.ceil(data[ylabel].mean() + 3*data[ylabel].std())
    xmax = np.max(data[xlabel])
    xmin = np.min(data[xlabel])

    x_ticks = np.arange(0, 0.4, 0.05)
    for wax in wue_plot.axes.ravel():
        wax.xaxis.set_ticks(x_ticks)
        wax.xaxis.set_ticklabels(['%1.2f' %x for x in x_ticks], \
                                 rotation=45, ha="right", fontsize=10)

    wue_plot.set(xlim=(xmin, xmax), ylim=(0, ymax))

    leg = plt.legend(loc='right', labels=month_lab, ncol=1, bbox_to_anchor=(1.3, 0.5), \
                     borderpad=2)
    leg.get_frame().set_edgecolor('black')

    wue_plot.fig.subplots_adjust(right=0.8, wspace=.08, hspace=0.15, top=0.9, bottom=0.25)

    return wue_plot


if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    INFILE = FILEPATH + "daily_inputs.pkl"
    OBSFILE = FILEPATH + "daily_tower_fluxes.pkl"
    MODFILE = FILEPATH + "daily_fluxes.pkl"

    # Figure names
    FIGPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/")

    main()



#    plt.savefig(WUEPLOT)
#
#    sns.set_style("darkgrid")
#    wue_plot = sns.FacetGrid(temp, col="Year", hue="Month", col_wrap=4, size=3)
#    wue_plot.map(plt.scatter, "Qle", "GPP2", **kws)
#    wue_plot.set_axis_labels('LE (MJ m$^{-2}$ d$^{-1}$)', 'GPP (gC m$^{-2}$ d$^{-1}$)')
#    wue_plot.set(xlim=(0, 20), ylim=(0, 10))
#    wue_plot.fig.subplots_adjust(wspace=.08)
#
#    plt.show()
#
#    return 1

