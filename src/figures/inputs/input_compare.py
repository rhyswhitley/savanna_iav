#!/usr/bin/env python2

import os
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import integrate
from matplotlib import style
#from mpltools import style

def plot_inputs(dataset, EX=1):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = plt.rcParams['axes.color_cycle']

    n_plots = 6
    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(n_plots, 1)
    ax1 = [plt.subplot(gs[i]) for i in range(n_plots)]

    # turn off x labels
    for i in range(5):
        ax1[i].xaxis.set_ticklabels([])

    date_ticks = pd.date_range("2001", periods=15, freq='AS')
    time_x = dataset.index
    # plots
    ax1[0].plot_date(time_x, dataset["SWdown"], '-', c=ncols[0], alpha=0.5, \
                     label="Tower")
    ax1[0].plot_date(time_x, dataset["clim_SWdown"], '-', c=ncols[3], alpha=1, \
                     label="Climatology")
    ax1[1].plot_date(time_x, dataset["VPD"], '-', c=ncols[0], alpha=0.5)
    ax1[1].plot_date(time_x, dataset["clim_VPD"], '-', c=ncols[3], alpha=1)
    ax1[2].plot_date(time_x, dataset["Tair"], '-', c=ncols[0], alpha=0.5)
    ax1[2].plot_date(time_x, dataset["clim_Tair"], '-', c=ncols[3], alpha=1)
    ax1[3].plot_date(time_x, dataset["Wind"], '-', c=ncols[0], alpha=0.5)
    ax1[3].plot_date(time_x, dataset["clim_Wind"], '-', c=ncols[3], alpha=1)
    ax1[4].bar(time_x, dataset["Rainfall"], color=None, edgecolor=ncols[0], alpha=0.3)
    ax1[4].bar(time_x, dataset["clim_Rainfall"], color=None, edgecolor=ncols[3], alpha=0.2)
    ax1[4].xaxis_date()
    ax2 = ax1[4].twinx()
    ax2.plot_date(time_x, dataset["Cair"], '-', c=ncols[0], lw=2.5, alpha=0.5)
    ax2.plot_date(time_x, dataset["clim_Cair"], '-', c=ncols[3], lw=2.5, alpha=1)
    ax1[5].plot_date(time_x, dataset["LAI"], '-', c=ncols[0], lw=2, alpha=0.7)
    ax1[5].plot_date(time_x, dataset["clim_LAI"], '-', c=ncols[3], alpha=1)

    # labels
    plt_label = "Howard Springs Meteorology 2001 to 2015"
    ax1[0].set_title(plt_label)
    ax1[0].set_ylabel("$R_{s}$ (MJ m$^{-2}$)")
    ax1[1].set_ylabel("$VPD$ (kPa)")
    ax1[2].set_ylabel("$T_{a}$ ($\degree$C)")
    ax1[3].set_ylabel("$v_{wind}$ (m s$^{-1}$)")
    ax1[4].set_ylabel("$PPT$ (mm)")
    ax1[5].set_ylabel("$LAI$ (m$^{2}$ m$^{-2}$)")
    ax2.set_ylabel("[$CO_{2}$] (ppm)")
    for i in range(len(ax1)):
        ax1[i].yaxis.set_label_coords(-0.07, 0.5)

    # limits
    ax1[0].set_ylim([10, 30])
    ax1[5].set_ylim([0.5, 2.5])
    ax2.set_ylim([350, 400])

    # axis
    ax2.grid(False)
    for i in range(len(ax1)):
        ax1[i].set_xticks(date_ticks)
    ax1[5].xaxis.set_ticklabels(date_ticks, rotation=45, ha="center", fontsize=11)
    ax1[5].xaxis.set_major_formatter(dates.DateFormatter('%Y'))

    # legends
    hand_1, labs_1 = ax1[0].get_legend_handles_labels()
    #hand_2, labs_2 = ax2.get_legend_handles_labels()
    ax1[5].legend(hand_1, labs_1, bbox_to_anchor=(0.5, -0.4), \
                  loc='upper center', ncol=4, prop={'size':10})

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.15)
    plt.savefig(SAVEPATH + "HWS_IAV_Inputs.pdf", rasterized=False)

    return fig

def get_value(nc_obj, label):
    return nc_obj.variables[label][:].flatten()

def get_dataframe(ncdf_con):
    """
    A quick function to transform a netcdf file into a pandas dataframe that
    can be used for analysis and plotting. Attributes are extracted using
    in built netCDF4 library functions. Time is arbitrary and needs to be
    set by the user.
    """
    # number of rows, equivalent to time-steps
    time_len = len(ncdf_con.dimensions['time'])
    # the header values for each measurements; excludes time and space components
    data_values = ncdf_con.variables.keys()[5:]

    # create a new dataframe from the netCDF file
    nc_dataframe = pd.DataFrame({label: get_value(ncdf_con, label) \
                                for label in data_values}, \
                                index=pd.date_range("2001-01-01 00:30:00", \
                                periods=time_len, freq="30min"))
    return nc_dataframe

def main():

    # make a connection to the netCDF file
    nc_conn = nc.Dataset(DIRPATH, 'r', format="NETCDF4")
    # transform netCDF file into a useable dataframe
    nc_data = get_dataframe(nc_conn)

    # reample this to daily, 7-day running mean
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    # sample functions
    samplers = [np.mean]*2 + [np.sum, dayint] + [np.mean]*3
    # dictionary to pass to resample
    sample_dict = {lab: samp for (lab, samp) in zip(nc_data.columns, samplers*2)}

    # downsampled to daily time-scale
    daily_data = nc_data.resample("D", how=sample_dict)

    # create a plot of the inputs
    rolling = lambda x: pd.rolling_mean(x, window=14, min_periods=1)
    plot_inputs(daily_data.apply(rolling, axis=0))

    return None

if __name__ == "__main__":

    DIRPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/spa_hws_inputs.nc")
    SAVEPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/inputs/")

    main()
