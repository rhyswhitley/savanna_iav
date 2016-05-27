#!/usr/bin/env python2

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import seaborn as sns
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
            sig_lw = 2.5
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
    data_values = ["Rainfall", "clim_Rainfall"]

    # create a new dataframe from the netCDF file
    nc_dataframe = pd.DataFrame({label: get_value(ncdf_con, label) \
                                for label in data_values}, \
                                index=pd.date_range("2001-01-01 00:30:00", \
                                periods=time_len, freq="30min"))
    return nc_dataframe

def plot_rainfall():

    # make a connection to the netCDF file
    nc_conn = nc.Dataset(FILEPATH, 'r', format="NETCDF4")
    # transform netCDF file into a useable dataframe
    nc_data = get_dataframe(nc_conn)

    annual_rain = nc_data.resample('AS', 'sum').ix[:-1, :]

    fig = plt.figure(figsize=(7, 5))
    grid = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(grid[0])

    p1, = ax1.plot(annual_rain.index, annual_rain.Rainfall, 'o--', alpha=0.7, label='Tower Observation')
    p2, = ax1.plot(annual_rain.index, annual_rain.clim_Rainfall, 'o-', label='Climatology')
    add_trend(ax1, annual_rain.index, annual_rain.Rainfall, p1.get_color(), 'Tower Trend')

    clim_mean = annual_rain.clim_Rainfall.mean()
    vary_mean = annual_rain.Rainfall.mean()
#    ax1.axhline(vary_mean, c='cyan')
#    ax1.annotate('Mean PPT is {0:.0f} mm'.format(vary_mean), fontsize=10, \
#                 xy=("2003-01-01", vary_mean), xytext=("2003-01-01", 500), \
#                    arrowprops=dict(facecolor='cyan', edgecolor='none', shrink=0.05))

    ax1.annotate('Mean PPT is {0:.0f} mm'.format(clim_mean), fontsize=10, \
                 xy=("2010-01-01", clim_mean), xytext=("2010-01-01", 500), \
                    arrowprops=dict(facecolor=p2.get_color(), edgecolor='none', shrink=0.05))

    ax1.set_ylabel("Annual Rainfall (mm)", fontsize=12)
    ax1.set_title("Howard Springs:: trends in rainfall", fontsize=13)

    newax1 = pd.date_range("2001", periods=14, freq='AS')
    ax1.xaxis.set_ticks(newax1)
    ax1.xaxis.set_ticklabels(newax1, rotation=45, ha="right", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.legend(loc='upper left', ncol=1)

    plt.ylim([0, 3000])

    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.98)
    plt.show()

    return 1

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/spa_hws_inputs.nc")

    plot_rainfall()
