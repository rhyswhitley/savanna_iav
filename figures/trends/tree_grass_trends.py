#!/usr/bin/env python2

from os.path import expanduser
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pylab import get_cmap


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
        pObj.legend(handles[part::2], labels[part::2], bbox_to_anchor=(xloc, yloc), \
                    prop={'size':fontsize}, loc='center', title=title)

def main():

    # retrieve SPA output files
    with open(FILEPATH, 'rb') as handle:
        canopy_data = pickle.load(handle)

    mol2g = 1/12
    s2d = 86400*1e-6
    col1_map = get_cmap("winter")(np.linspace(0, 1, len(canopy_data)))
    col2_map = get_cmap("autumn")(np.linspace(0, 1, len(canopy_data)))

    # create plotting area
    plt.figure(figsize=(10, 8))
    plot_grid = gridspec.GridSpec(2, 1, hspace=0.1)
    # create subplots
    ax1 = plt.subplot(plot_grid[0])
    ax2 = plt.subplot(plot_grid[1])

    # for each experiment
    for (i, (clab, hr_can)) in enumerate(canopy_data.iteritems()):

        # resample to day timestep
        day_can = hr_can.ix[:, ["trees_Ag", "trees_Et", "grass_Ag", "grass_Et"]] \
                    .resample('D', how=lambda x: np.mean(x)*24)
        year_can = day_can.groupby([day_can.index.year]) \
                    .aggregate("sum")
        # time
        d_tseries = day_can.index
        y_tseries = range(2001, 2015)

        # water
        ax1.plot(y_tseries, year_can["trees_Et"], 'o--', alpha=0.4, c=col1_map[i])
        ax1.plot(y_tseries, year_can["grass_Et"], 'o--', alpha=0.4, c=col2_map[i])
        # y ~ x
        add_trend(ax1, y_tseries, year_can["trees_Et"], col1_map[i], clab)
        add_trend(ax1, y_tseries, year_can["grass_Et"], col2_map[i], clab)


        # carbon
        ax2.plot(y_tseries, year_can["trees_Ag"]/12, 'o--', alpha=0.4, c=col1_map[i])
        ax2.plot(y_tseries, year_can["grass_Ag"]/12, 'o--', alpha=0.4, c=col2_map[i])
        # y ~ x
        add_trend(ax2, y_tseries, year_can["trees_Ag"]/12, col1_map[i], clab)
        add_trend(ax2, y_tseries, year_can["grass_Ag"]/12, col2_map[i], clab)

        ax1.set_xlim([2000, 2015])
        ax2.set_xlim([2000, 2015])

        add_mylegend(ax1, 0, xloc=1.1, yloc=0.75, fontsize=8, title="Trees")
        add_mylegend(ax2, 0, xloc=1.1, yloc=0.75, fontsize=8, title="Trees")

    plt.subplots_adjust(right=0.8)
    plt.show()
    return 1

if __name__ == "__main__":

    FILEPATH = expanduser("~/Savanna/Data/HowardSprings_IAV/gasex/spa_treegrass_reduced.pkl")

    main()
