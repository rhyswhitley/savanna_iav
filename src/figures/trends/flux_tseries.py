#!/usr/bin/env python

import os
import cPickle as pickle
import pandas as pd
import numpy as np
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style

def plot_flux_diff(df_0, df_x, ex_0, ex_x):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'grid.alpha': 0.7})
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = np.array(plt.rcParams['axes.color_cycle'])[[0, 1, 5]]
    bcols = [u'#8B0000', u'#00008B', u'#006400']

    # setup plotting canvas
    fig = plt.figure(figsize=(11, 6))
    grid = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(grid[0])
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    # smooth the t-series using a rolling-mean
    rolling = lambda x: pd.rolling_mean(x, window=14, min_periods=1)
    df_0r7 = df_0.apply(rolling, axis=0)
    df_xr7 = df_x.apply(rolling, axis=0)

    # x-axis vector
    x_time = df_0.index

    labplot = lambda x, y: '{0} [Exp. {1}]'.format(x, y)

    # set to variables
    convert = 12
    gpp_0r7 = [min(xi*convert, -2) for xi in df_0r7["GPP"].values]
    gpp_xr7 = [min(xi*convert, -2) for xi in df_xr7["GPP"].values]
    qle_0r7 = df_0r7["Qle"]
    qle_xr7 = df_xr7["Qle"]

    # plot latent energy
    ax1.plot_date(x_time, qle_0r7, '-', c=bcols[0], label=labplot("LE", ex_0))
    ax1.plot_date(x_time, qle_xr7, '-', c=ncols[0], label=labplot("LE", ex_x))

    # plot gross primary productivity
    ax1.plot_date(x_time, gpp_0r7, '-', c=bcols[1], label=labplot("GPP", ex_0))
    ax1.plot_date(x_time, gpp_xr7, '-', c=ncols[1], label=labplot("GPP", ex_x))

    # plot water-use efficiency
    ax2.plot_date(x_time, -qle_0r7/gpp_0r7, '-', c=bcols[2], label=labplot("WUE", ex_0))
    ax2.plot_date(x_time, -qle_xr7/gpp_xr7, '-', c=ncols[2], label=labplot("WUE", ex_x))

    # limits
    ax1.set_ylim([-10, 15])
    ax2.set_ylim([-1.5, 6])

    # labels
    ax1.set_ylabel(r'LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=12)
    ax1.yaxis.set_label_coords(-0.05, 0.7)

    ax2.set_ylabel(r'WUE (MJ gC$^{-1}$)', y=0.45, x=0.1, fontsize=12)
    ax2.yaxis.set_label_position("right")

    ax3.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=12)
    ax3.yaxis.set_label_coords(-0.08, 0.2)


    # axis
    flux_ticks = np.arange(-10, 16, 2)
    ax1.yaxis.set_ticks(flux_ticks)
    ax1.yaxis.set_ticklabels(flux_ticks, fontsize=11)

    date_ticks = pd.date_range("2001", "2015", freq='AS')
    ax1.xaxis.set_ticks(date_ticks)
    ax1.xaxis.set_ticklabels(date_ticks, rotation=45, ha="center", fontsize=12)
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%Y'))

    wue_ticks = np.arange(0, 4, 0.5)
    ax2.yaxis.set_ticks(wue_ticks)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticklabels(wue_ticks, fontsize=11)

    ax3.yaxis.set_ticks([])

    ax2.grid(False)

    # title
    ax1.set_title("Howard Springs (2001 - 2015) :: Simluated Water and Carbon Cycling", y=1.03)

    # legends
    hand1, labs1 = ax1.get_legend_handles_labels()
    hand2, labs2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(hand1+hand2, labs1+labs2, bbox_to_anchor=(0.5, -0.12), \
                  loc='upper center', ncol=6, prop={'size':10})
    for leg in leg.get_lines():
        leg.set_alpha(1)
        leg.set_lw(3)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.15, hspace=0.15)

    # saving
    save_file = "Experiment{0}_flux_tseries.pdf".format(ex_x)
    plt.savefig(SAVEDIR + save_file)
    #plt.show()

    return fig

def main():

    flux_dict = pickle.load(open(PKLPATH, 'rb'))
    flux_list = flux_dict.values()

    exp_3 = plot_flux_diff(flux_dict['Exp_2'], flux_dict['Exp_3'], 2, 3)
    exp_14 = plot_flux_diff(flux_dict['Exp_1'], flux_dict['Exp_14'], 1, 14)

    return 1

if __name__ == "__main__":

    PKLDIR = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    PKLFILE = "daily_fluxes.pkl"
    PKLPATH = PKLDIR + PKLFILE

    # save here
    SAVEDIR = os.path.expanduser("~/Savanna/Analysis/figures/IAV/tseries/")

    main()
