#!/usr/bin/env python2

import os, re
import seaborn as sns
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy import stats

def fit_trend(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {'model': p, 'slope': slope, 'int': intercept, \
            'r2': r_value, 'se': std_err, 'p': p_value}

def add_trend(pObj, xseries, yseries, col, header):
        # get linear trend line
        xtrans = mdates.date2num(xseries.to_pydatetime())
        trend = fit_trend(xtrans, yseries)
        # make the significant slopes stand out more
        if trend['p'] < 0.05:
            sig_lw = 2
            sig_alpha = 1
        else:
            sig_lw = 1.5
            sig_alpha = 0.5
        # create a label to shows statistics
        trend_lab = '{0}: slope = {1:.2f}, p = {2:.3f}' \
            .format(header, trend['slope'], trend['p'])
        # plot trend line
        pObj.plot(xseries, trend['model'](xtrans), '-', c=col, \
                  label=trend_lab, alpha=sig_alpha, lw=sig_lw)

def main():
    """
    Does a quick re-sampling of flux tower observation form hourly to daily
    timesteps
    """

    pload = lambda x: pickle.load(open(x, 'rb'))

    model_dict = pload(MODFILE)

    temp = model_dict['Exp_1']
    clim = model_dict['Exp_2']
    temp['Date'] = temp.index
    clim['Date'] = clim.index
    temp['LAItot'] = temp['LAIgrass'] + temp['LAItree']
    clim['LAItot'] = clim['LAIgrass'] + clim['LAItree']

    #lai_cols = [tc for tc in temp.columns if re.search(r'^LAI.*', tc)]
    sns.set_style("whitegrid")
    plt.rcParams.update({'mathtext.default': 'regular'})
    plt.rcParams.update({'grid.alpha': 0.3})

    fig = plt.figure(figsize=(9, 6))

    grid = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(grid[0])
    ax2 = plt.subplot(grid[1])

    # top plot
    p1, = ax1.plot(temp['LAItot'], alpha=0.4)
    p2, = ax1.plot(temp['LAItree'], alpha=0.4)
    p3, = ax1.plot(temp['LAIgrass'], alpha=0.4)
    add_trend(ax1, temp.index, temp['LAItot'], p1.get_color(), 'Total')
    add_trend(ax1, temp.index, temp['LAItree'], p2.get_color(), 'Tree')
    add_trend(ax1, temp.index, temp['LAIgrass'], p3.get_color(), 'Grass')

    # bottom plot
    vary_mean = temp['LAItot'].mean()
    clim_mean = clim['LAItot'].mean()
    ax2.plot(temp['LAItot'], '--', lw=1.2, alpha=0.6, c=p1.get_color(), label='Variable LAI')
    ax2.axhline(vary_mean, c=p1.get_color(), label='Mean of Variable')
    ax2.plot(clim['LAItot'], '--', lw=1.2, alpha=0.6, c=p3.get_color(), label='Repeated LAI')
    ax2.axhline(clim_mean, c=p3.get_color(), label='Mean of Repeated')

    # labels
    ax1.set_ylabel("Leaf Area Index (m$^{2}$ m$^{-2}$)", fontsize=12, y=-0.05)
    ax1.set_title("Howard Springs:: [no] trends in LAI", fontsize=13)
    ax1.set_ylim([0, 2.8])
    ax2.set_ylim([0, 2.8])
    # means
    ax2.annotate('Mean is {0:.2f}'.format(vary_mean), fontsize=10, \
                 xy=("2002-01-01", vary_mean), xytext=("2002-01-01", 0.5), \
                    arrowprops=dict(facecolor=p1.get_color(), edgecolor='none', shrink=0.05))
    ax2.annotate('Mean is {0:.2f}'.format(clim_mean), fontsize=10, \
                 xy=("2004-01-01", clim_mean), xytext=("2004-01-01", 0.5), \
                    arrowprops=dict(facecolor=p3.get_color(), edgecolor='none', shrink=0.05))

    # axis ticks
    ax1.tick_params(labelsize=11)
    ax2.tick_params(labelsize=11)
    newax1 = pd.date_range("2001", periods=15, freq='AS')
    ax1.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticks(newax1)
    ax2.xaxis.set_ticks(newax1)
    ax2.xaxis.set_ticklabels(newax1, rotation=45, ha="right", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[3:], labels=labels[3:], loc='upper center', ncol=3)
    ax2.legend(loc='upper center', ncol=4)
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.98, hspace=0.1)

    plt.savefig(LAIPLOT)
    #plt.show()
    return 1

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    INFILE = FILEPATH + "daily_inputs.pkl"
    OBSFILE = FILEPATH + "daily_tower_fluxes.pkl"
    MODFILE = FILEPATH + "daily_leaf.pkl"

    # Figure names
    FIGPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/")
    LAIPLOT = FIGPATH + "lai_trends.pdf"

    main()
