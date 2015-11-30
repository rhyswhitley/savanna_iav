#!/usr/bin/env python

import os
import cPickle as pickle
import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style

def plot_flux_diff(df_0, df_x, ex_0, ex_x):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    #ncols = plt.rcParams['axes.color_cycle']

    # setup plotting canvas
    fig = plt.figure(figsize=(8, 6))
    grid = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1])

    # smooth the t-series using a rolling-mean
    rolling = lambda x: pd.rolling_mean(x, window=14, min_periods=1)
    df_0r7 = df_0.apply(rolling, axis=0)
    df_xr7 = df_x.apply(rolling, axis=0)

    # x-axis vector
    x_time = df_0.index

    # plot bot
    ax1.plot_date(x_time, df_0r7["Qle"], '-')
    ax1.plot_date(x_time, df_xr7["Qle"], '-')

    # plot top
    convert = 12
    labplot = lambda x: 'Experiment {0}'.format(x)
    ax2.plot_date(x_time, df_0r7["GPP"]*convert, '-', label=labplot(ex_0))
    ax2.plot_date(x_time, df_xr7["GPP"]*convert, '-', label=labplot(ex_x))

    # limits
    ax1.set_ylim([0, 16])
    ax2.set_ylim([-10, 0])

    # labels
    ax1.set_ylabel(r'LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=11)
    ax2.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=11)

    # axis
    date_ticks = pd.date_range("2001", "2015", freq='AS')
    ax1.xaxis.set_ticks(date_ticks)
    ax2.xaxis.set_ticks(date_ticks)
    ax1.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels(date_ticks, rotation=45, ha="center", fontsize=11)
    ax2.xaxis.set_major_formatter(dates.DateFormatter('%Y'))

    ax1.grid(color='gray')
    ax2.grid(color='gray')

    # legends
    ax2.legend(bbox_to_anchor=(0.5, -0.23), \
                  loc='upper center', ncol=2, prop={'size':10})

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, hspace=0.15)

    # saving
    save_file = "FluxCompares_Exps{0}-{1}.pdf".format(ex_0, ex_x)
    plt.savefig(SAVEDIR + save_file)

def main():

    flux_dict = pickle.load(open(PKLPATH, 'rb'))
    flux_list = flux_dict.values()
    myfigs = [plot_flux_diff(flux_list[0], flux_list[i], 1, i + 1) \
                for i in range(1, len(flux_list)) if i < 100]

    return 1

if __name__ == "__main__":

    PKLDIR = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    PKLFILE = "daily_fluxes.pkl"
    PKLPATH = PKLDIR + PKLFILE

    # save here
    SAVEDIR = os.path.expanduser("~/Savanna/Analysis/figures/IAV/tseries/")

    main()
