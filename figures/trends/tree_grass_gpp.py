#!/usr/bin/env python2

from os.path import expanduser
import collections
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pylab import get_cmap
from scipy import integrate


def fit_trend(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {'model': p, 'slope': slope, 'int': intercept, \
            'r2': r_value, 'se': std_err, 'p': p_value}

def add_trend(pObj, xseries, yseries, col, header):
        trend = fit_trend(xseries, yseries)
        trend_lab = '{0}: slope = {1:.2f}, p = {2:.3f}' \
            .format(header, trend['slope'], trend['p'])
        pObj.plot(xseries, trend['model'](xseries), '-', c=col, label=trend_lab, alpha=1, lw=2)

def add_mylegend(pObj, part, title, fontsize=11, xloc=0, yloc=0):
        handles, labels = pObj.get_legend_handles_labels()
        leg = pObj.legend(handles[part::2], labels[part::2], bbox_to_anchor=(xloc, yloc), \
                    prop={'size':fontsize}, loc='center', title=title)
        return leg

def main():

    # retrieve SPA output files
    with open(FILENAME1, 'rb') as handle:
        canopy_data_0 = pickle.load(handle)

    canopy_data = collections.OrderedDict(sorted(canopy_data_0.iteritems()))

    col1_map = get_cmap("winter")(np.linspace(0, 1, len(canopy_data)))
    col2_map = get_cmap("autumn")(np.linspace(0, 1, len(canopy_data)))

    # create plotting area
    plt.figure(figsize=(10, 6))
    plot_grid = gridspec.GridSpec(1, 1, hspace=0.1)
    # create subplots
    ax1 = plt.subplot(plot_grid[0])

    # for each experiment
    for (i, (clab, hr_can)) in enumerate(canopy_data.iteritems()):

        # resample to day timestep
        day_can = hr_can.ix[:, ["trees_Ag", "trees_Et", "grass_Ag", "grass_Et"]] \
                    .resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6*12)
        year_can = day_can.groupby([day_can.index.year]) \
                    .aggregate("sum")
        # time
        d_tseries = day_can.index
        y_tseries = range(2001, 2015)

        # carbon
        ax1.plot(y_tseries, year_can["trees_Ag"], 'o--', alpha=0.4, c=col1_map[i])
        ax1.plot(y_tseries, year_can["grass_Ag"], 'o--', alpha=0.4, c=col2_map[i])
        # y ~ x
        add_trend(ax1, y_tseries, year_can["trees_Ag"], col1_map[i], clab)
        add_trend(ax1, y_tseries, year_can["grass_Ag"], col2_map[i], clab)

        ax1.set_xlim([2000, 2015])
        newax = np.arange(2001, 2015, 1)
        ax1.xaxis.set_ticks(newax)
        ax1.xaxis.set_ticklabels(newax, rotation=45, ha="right", fontsize=13)

        leg1 = add_mylegend(ax1, 0, xloc=1.15, yloc=0.75, fontsize=9, title="Trees")
        add_mylegend(ax1, 1, xloc=1.15, yloc=0.25, fontsize=9, title="Grass")
        plt.gca().add_artist(leg1)

    plt.title("Howard Springs IAV Experiments 2001 to 2015")
    plt.ylabel(r"Gross Primary Productivity (gC m$^{-2}$ s$^{-1}$)")

    plt.grid(c='gray')

    plt.subplots_adjust(left=0.1, right=0.76, bottom=0.1, top=0.95)
    plt.savefig(SAVEPATH)
    return 1

if __name__ == "__main__":

    FILEPATH = expanduser("~/Savanna/Data/HowardSprings_IAV/")
    FILENAME1 = FILEPATH + "gasex/spa_treegrass_reduced.pkl"
    FILENAME2 = FILEPATH + "bulkflux/spa_hourly_output.pkl"

    SAVEPATH = expanduser("~/Savanna/Analysis/figures/IAV/HWS_treegrass_gpptrends.pdf")

    main()
