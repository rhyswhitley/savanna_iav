#!/usr/bin/env python2

import os
import re
import natsort
import netCDF4 as nc
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib import style
from scipy import stats
from collections import OrderedDict

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
        # create a label to shows statistics
        trend_lab = '{0}: slope = {1:.2f}, p = {2:.3f}' \
            .format(header, trend['slope'], trend['p'])
        # make the significant slopes stand out more
        if trend['p'] < 0.05:
            sig_lw = 4
            sig_alpha = 1
        else:
            sig_lw = 1.75
            sig_alpha = 0.5
        # plot trend line
        pObj.plot(xseries, trend['model'](xtrans), '-', c=col, \
                  label=trend_lab, alpha=sig_alpha, lw=sig_lw)

def add_mylegend(pObj, part, title, step=2, fontsize=11, xloc=0, yloc=0):
        handles, labels = pObj.get_legend_handles_labels()
        leg = pObj.legend(handles[part::step], labels[part::step], bbox_to_anchor=(xloc, yloc), \
                    prop={'size':fontsize}, loc='center', title=title)
        return leg

def agg_tseries_up(init_df, conv=1):

    # upscale to day :: integrated sum [response to discrete value]
    day_df = init_df.resample('D', how= \
                lambda x: integrate.trapz(x, dx=1800)*conv)

    # upscale to month: simple sum
    month_df = day_df.groupby([day_df.index.year, day_df.index.month]) \
                .aggregate("sum")

    # upscale to year: simple sum
    year_df = day_df.groupby([day_df.index.year]).aggregate("sum")

    # return to user the time-series at different levels
    return {"day": day_df, "month": month_df, "year": year_df}

def get_value(nc_obj, label):
    return nc_obj.variables[label][:].flatten()

def get_dataframe(npath):
    """
    A quick function to transform a netcdf file into a pandas dataframe that
    can be used for analysis and plotting. Attributes are extracted using
    in built netCDF4 library functions. Time is arbitrary and needs to be
    set by the user.
    """
    # make a connection to the ncdf file
    ncdf_con = nc.Dataset(npath, 'r', format="NETCDF4")
    # number of rows, equivalent to time-steps
    time_len = len(ncdf_con.dimensions['time'])
    # extract time information
    time_sec = ncdf_con.variables['time']
    sec_orig = re.search(r'\d+.*', str(time_sec.units)).group(0)

    # create a new dataframe from the netCDF file
    nc_dataframe = pd.DataFrame({"trees_Ag": get_value(ncdf_con, "Atree"), \
                                "grass_Ag": get_value(ncdf_con, "Agrass")}, \
                                index=pd.date_range(sec_orig, \
                                periods=time_len, freq="30min"))
    return nc_dataframe

def plot_gpp_trends(canopy_data_list):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})

    n_exps = len(canopy_data_list)
    col1_map = get_cmap("summer")(np.linspace(0.1, 1, n_exps))
    col2_map = get_cmap("winter")(np.linspace(0.1, 1, n_exps))
    col3_map = get_cmap("autumn")(np.linspace(0.1, 1, n_exps))

    # create plotting area
    plt.figure(figsize=(10, 9))
    plot_grid = gridspec.GridSpec(1, 1, hspace=0.1)
    # create subplots
    ax1 = plt.subplot(plot_grid[0])

    # for each experiment
    for (i, canopy) in enumerate(canopy_data_list):

        # resample to day timestep
        canp_upts = agg_tseries_up(canopy.ix[:, ["trees_Ag", "grass_Ag"]], 1e-6*12)

        # time
        trees_Cf = np.array(canp_upts["year"]["trees_Ag"])
        grass_Cf = np.array(canp_upts["year"]["grass_Ag"])
        total_Cf = trees_Cf + grass_Cf
        y_tseries = np.arange(2001, 2015)
        #import ipdb; ipdb.set_trace()

        # carbon
        ax1.plot(y_tseries, total_Cf, 'o--', alpha=0.4, c=col1_map[i])
        ax1.plot(y_tseries, trees_Cf, 'o--', alpha=0.4, c=col2_map[i])
        ax1.plot(y_tseries, grass_Cf, 'o--', alpha=0.4, c=col3_map[i])
        # y ~ x
        add_trend(ax1, y_tseries, total_Cf, col1_map[i], "Exp_{0}".format(i+1))
        add_trend(ax1, y_tseries, trees_Cf, col2_map[i], "Exp_{0}".format(i+1))
        add_trend(ax1, y_tseries, grass_Cf, col3_map[i], "Exp_{0}".format(i+1))

    # limits
    ax1.set_ylim([100, 1800])
    ax1.set_xlim([2000, 2015])

    # axis
    newXax = np.arange(2001, 2015, 1)
    newYax = np.arange(200, 1800, 100)
    ax1.xaxis.set_ticks(newXax)
    ax1.yaxis.set_ticks(newYax)
    ax1.xaxis.set_ticklabels(newXax, rotation=45, ha="right", fontsize=13)
    ax1.yaxis.set_ticklabels(newYax, fontsize=13)

    # labels
    ax1.set_title("Howard Springs IAV Experiments (2001 to 2015)")
    ax1.set_ylabel(r"Gross Primary Productivity (gC m$^{-2}$ s$^{-1}$)")
    ax1.yaxis.set_label_coords(-0.1, 0.5)

    # legendsz
    leg1 = add_mylegend(ax1, part=0, title="Total", step=3, xloc=1.15, yloc=0.85, fontsize=9)
    leg2 = add_mylegend(ax1, part=1, title="Tree", step=3, xloc=1.15, yloc=0.63, fontsize=9)
    leg3 = add_mylegend(ax1, part=2, title="Grass", step=3, xloc=1.15, yloc=0.13, fontsize=9)
    plt.gca().add_artist(leg1)
    plt.gca().add_artist(leg2)
    plt.gca().add_artist(leg3)

    ax1.grid(c='gray')

    plt.subplots_adjust(left=0.1, right=0.76, bottom=0.1, top=0.95)
    #plt.savefig(SAVEPATH)
    plt.show()
    return 1

def plot_gpp_trends_split(canopy_data_list, sname, lai_var=True):

    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams.update({'mathtext.default': 'regular'})

    # create plotting area
    plt.figure(figsize=(10, 9))
    plot_grid = gridspec.GridSpec(3, 1, hspace=0.1)
    # create subplots
    ax1 = plt.subplot(plot_grid[0])
    ax2 = plt.subplot(plot_grid[1])
    ax3 = plt.subplot(plot_grid[2])

    # for each experiment
    for (i, (clab, canopy)) in enumerate(canopy_data_list.iteritems()):

        if lai_var is True:
            take_these = [0] + range(7, 13)
        else:
            take_these = range(1, 7) + [13]

        if i in take_these:
            # get values
            trees_Cf = canopy["Atree"].values
            grass_Cf = canopy["Agrass"].values
            total_Cf = trees_Cf + grass_Cf
            y_tseries = canopy.index

            # carbon
            p1, = ax1.plot(y_tseries, total_Cf, '-', alpha=0.4)
            p2, = ax2.plot(y_tseries, trees_Cf, '-', alpha=0.4)
            p3, = ax3.plot(y_tseries, grass_Cf, '-', alpha=0.4)
            # y ~ x
            add_trend(ax1, y_tseries, total_Cf, p1.get_color(), clab)
            add_trend(ax2, y_tseries, trees_Cf, p2.get_color(), clab)
            add_trend(ax3, y_tseries, grass_Cf, p3.get_color(), clab)

#    # limits
#    ax1.set_xlim([2000, 2015])
#    ax2.set_xlim([2000, 2015])
#    ax3.set_xlim([2000, 2015])
#
#    # axis
#    newax = pd.date_range(start="2001-01-01", end="20015-01-01", freq='AS')
#    ax1.xaxis.set_ticks(newax)
#    ax2.xaxis.set_ticks(newax)
#    ax3.xaxis.set_ticks(newax)
#    ax1.xaxis.set_ticklabels([])
#    ax2.xaxis.set_ticklabels([])
#    ax3.xaxis.set_ticklabels(newax, rotation=45, ha="right", fontsize=13)

    # labels
    if lai_var is True:
        title_lab = "Howard Springs [2001 - 2015] :: vary-IAV LAI Experiments"
    else:
        title_lab = "Howard Springs [2001 - 2015] :: non-IAV LAI Experiments"

    ax1.set_title(title_lab)
    ax1.set_ylabel(r"Total GPP (gC m$^{-2}$ s$^{-1}$)")
    ax2.set_ylabel(r"Tree GPP (gC m$^{-2}$ s$^{-1}$)")
    ax3.set_ylabel(r"Grass GPP (gC m$^{-2}$ s$^{-1}$)")
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    ax3.yaxis.set_label_coords(-0.1, 0.5)

    # legends
    ax1.legend(loc="right", bbox_to_anchor=(1.35, 0.5), prop={'size':9}, ncol=1)
    ax2.legend(loc="right", bbox_to_anchor=(1.35, 0.5), prop={'size':9}, ncol=1)
    ax3.legend(loc="right", bbox_to_anchor=(1.35, 0.5), prop={'size':9}, ncol=1)

    #ax1.grid(c='gray')
    #ax2.grid(c='gray')
    #ax3.grid(c='gray')

    plt.subplots_adjust(left=0.1, right=0.76, bottom=0.1, top=0.95)
    #plt.savefig(SAVEPATH + sname)
    plt.show()
    return 1


def main():

    # Get the leaf daily dataframe dictionary
    leaf_dict = pickle.load(open(PKLPATH + "daily/daily_leaf.pkl", 'rb'))

    # for some reason the natural sorting isn't retained in the load
    leaf_year = OrderedDict(natsort.natsorted({dlab: ldf[["Atree", "Agrass"]] \
                    .resample("M", how=lambda x: sum(x)*12) \
                    for (dlab, ldf) in leaf_dict.iteritems()}.iteritems()))

    # Create the plot
    #plot_gpp_trends_split(leaf_year, "HWS_GPP_trend_nonLAI.pdf", lai_var=False)
    plot_gpp_trends_split(leaf_year, "HWS_GPP_trend_varLAI.pdf", lai_var=True)

    return None

if __name__ == "__main__":

    # set paths
    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")
    SAVEPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/")

    # run main
    main()



