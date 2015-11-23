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

def agg_tseries_up(df, labels, conv=1):

    # upscale to day :: integrated sum [response to discrete value]
    day_df = df.ix[:, labels].resample('D', how= \
                lambda x: integrate.trapz(x, dx=1800)*conv)

    # upscale to month: simple sum
    month_df = day_df.groupby([day_df.index.year, day_df.index.month]) \
                .aggregate("sum")

    # upscale to year: simple sum
    year_df = day_df.groupby([day_df.index.year]).aggregate("sum")

    # return to user the time-series at different levels
    return {"day": day_df, "month": month_df, "year": year_df}
def main():

    # retrieve SPA output files
    with open(FILENAME1, 'rb') as handle:
        canopy_data_0 = pickle.load(handle)
    with open(FILENAME2, 'rb') as handle:
        hourly_data_0 = pickle.load(handle)

    canopy_data = collections.OrderedDict(sorted(canopy_data_0.iteritems()))
    hourly_data = collections.OrderedDict(sorted(hourly_data_0.iteritems()))

    col1_map = get_cmap("gray")(np.linspace(0.05, 1, len(hourly_data)))
    col2_map = get_cmap("winter")(np.linspace(0.05, 1, len(canopy_data)))
    col3_map = get_cmap("autumn")(np.linspace(0.05, 1, len(canopy_data)))

    # create plotting area
    plt.figure(figsize=(10, 9))
    plot_grid = gridspec.GridSpec(3, 1, hspace=0.1)
    # create subplots
    ax1 = plt.subplot(plot_grid[0])
    ax2 = plt.subplot(plot_grid[1])
    ax3 = plt.subplot(plot_grid[2])

    # for each experiment
    for (i, ((clab, canopy), (hlab, hourly))) in enumerate( \
        zip(canopy_data.iteritems(), hourly_data.iteritems())):

        # resample to day timestep
        canp_upts = agg_tseries_up(canopy, ["trees_Ag", "grass_Ag"], 1e-6*12)
        hour_upts = agg_tseries_up(hourly, ["GPP", "resp"], 1e-6*12)

        # time
        y_tseries = range(2001, 2015)

        # carbon
        ax1.plot(y_tseries, -hour_upts["year"]["GPP"], 'o--', alpha=0.4, c=col1_map[i])
        ax2.plot(y_tseries, canp_upts["year"]["trees_Ag"], 'o--', alpha=0.4, c=col2_map[i])
        ax3.plot(y_tseries, canp_upts["year"]["grass_Ag"], 'o--', alpha=0.4, c=col3_map[i])
        # y ~ x
        add_trend(ax1, y_tseries, -hour_upts["year"]["GPP"], col1_map[i], hlab)
        add_trend(ax2, y_tseries, canp_upts["year"]["trees_Ag"], col2_map[i], clab)
        add_trend(ax3, y_tseries, canp_upts["year"]["grass_Ag"], col3_map[i], clab)

    # limits
    ax1.set_ylim([1600, 2100])
    ax2.set_ylim([1100, 1500])
    ax3.set_ylim([300, 600])

    ax1.set_xlim([2000, 2015])
    ax2.set_xlim([2000, 2015])
    ax3.set_xlim([2000, 2015])

    # axis
    newax = np.arange(2001, 2015, 1)
    ax1.xaxis.set_ticks(newax)
    ax2.xaxis.set_ticks(newax)
    ax3.xaxis.set_ticks(newax)
    ax1.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])
    ax3.xaxis.set_ticklabels(newax, rotation=45, ha="right", fontsize=13)

    # labels
    ax1.set_title("Howard Springs IAV Experiments (2001 to 2015)")
    ax1.set_ylabel(r"Total GPP (gC m$^{-2}$ s$^{-1}$)")
    ax2.set_ylabel(r"Tree GPP (gC m$^{-2}$ s$^{-1}$)")
    ax3.set_ylabel(r"Grass GPP (gC m$^{-2}$ s$^{-1}$)")
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    ax3.yaxis.set_label_coords(-0.1, 0.5)

    # legends
    ax1.legend(loc="right", bbox_to_anchor=(1.35, 0.5), \
                            prop={'size':9}, ncol=1)
    ax2.legend(loc="right", bbox_to_anchor=(1.35, 0.5), \
                            prop={'size':9}, ncol=1)
    ax3.legend(loc="right", bbox_to_anchor=(1.35, 0.5), \
                            prop={'size':9}, ncol=1)

    ax1.grid(c='gray')
    ax2.grid(c='gray')
    ax3.grid(c='gray')

    plt.subplots_adjust(left=0.1, right=0.76, bottom=0.1, top=0.95)
    #plt.savefig(SAVEPATH)
    plt.show()
    return 1

if __name__ == "__main__":

    FILEPATH = expanduser("~/Savanna/Data/HowardSprings_IAV/")
    FILENAME1 = FILEPATH + "gasex/spa_treegrass_reduced.pkl"
    FILENAME2 = FILEPATH + "bulkflux/spa_hourly_output.pkl"

    SAVEPATH = expanduser("~/Savanna/Analysis/figures/IAV/HWS_treegrass_gpptrends.pdf")

    main()


# old double legend on same plot
#        leg1 = add_mylegend(ax1, 0, xloc=1.15, yloc=0.75, fontsize=9, title="Trees")
#        add_mylegend(ax1, 1, xloc=1.15, yloc=0.25, fontsize=9, title="Grass")
#        plt.gca().add_artist(leg1)

