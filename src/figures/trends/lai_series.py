#!/usr/bin/env python2

import os, re
import seaborn as sns
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
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
            sig_lw = 4
            sig_alpha = 1
        else:
            sig_lw = 1.75
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
    temp['Date'] = temp.index
    temp['LAItot'] = temp['LAIgrass'] + temp['LAItree']

    #lai_cols = [tc for tc in temp.columns if re.search(r'^LAI.*', tc)]
    sns.set_style("whitegrid")
    plt.rcParams.update({'mathtext.default': 'regular'})
    plt.rcParams.update({'grid.alpha': 0.3})

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(1, 1, 1)

    p1, = ax.plot(temp['LAItot'], alpha=0.4)
    p2, = ax.plot(temp['LAItree'], alpha=0.4)
    p3, = ax.plot(temp['LAIgrass'], alpha=0.4)
    add_trend(ax, temp.index, temp['LAItot'], p1.get_color(), 'Total')
    add_trend(ax, temp.index, temp['LAItree'], p2.get_color(), 'Tree')
    add_trend(ax, temp.index, temp['LAIgrass'], p3.get_color(), 'Grass')

    # labels
    ax.set_ylabel("Leaf Area Index (m$^{2}$ m$^{-2}$)", fontsize=12)
    ax.set_title("Howard Springs [no] trends in LAI", fontsize=14)


    # axis ticks
    ax.tick_params(labelsize=11)
    newax = pd.date_range("2001", periods=15, freq='AS')
    ax.xaxis.set_ticks(newax)
    ax.xaxis.set_ticklabels(newax, rotation=45, ha="right", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles[3:], labels=labels[3:], loc='upper left')
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.92, right=0.98)

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
