#!/usr/bin/env python2

from os.path import expanduser
import collections
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pylab import get_cmap
from scipy import integrate


def import_spa_output(file_name, spaf="D"):
    """
    Imports dingo meteorology file and puts an index on the time column
    """
    # read in file
    data = pd.read_csv(file_name)
    data.columns = [dc.replace(' ', '') for dc in data.columns]
    # re-do the datetime column as it is out by 30 minutes
    data['DATE'] = pd.date_range(start="2001-01-01", periods=len(data), freq=spaf)
    # set dataframe index on this column
    data_dated = data.set_index(['DATE'])
    # return to user
    return data_dated

def main():

    # retrieve SPA output files
    exp_path1 = expanduser("~/Savanna/Models/SPA1/outputs/site_co2/HS_Exp1/outputs/")
    exp_path2 = expanduser("~/Savanna/Data/HowardSprings_IAV/")

    exp1_dy = import_spa_output(exp_path1 + "daily.csv", "D")
    exp1_tg = import_spa_output(exp_path2 + "gasex/exp_1-gasex_HWS.csv", "30min")
    exp1_hr = import_spa_output(exp_path2 + "bulkflux/exp_1-bulkflux_HWS.csv", "30min")

    day_can = exp1_tg.ix[:, ["trees_Ag", "grass_Ag"]] \
                .resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6*12)
    year_can = day_can.groupby([day_can.index.year]) \
                .aggregate("sum")

    day_blk = exp1_hr.ix[:, ["GPP"]] \
                .resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6*12)
    year_blk = day_blk.groupby([day_blk.index.year]) \
                .aggregate("sum")

    year_day = exp1_dy.groupby([exp1_dy.index.year]).aggregate("sum")

    tseries = exp1_dy.index
    plt.plot_date(tseries, exp1_dy['gpp'], '-')
    #plt.plot_date(tseries, -day_blk['GPP'], '-')
    #plt.plot_date(tseries, day_can['trees_Ag'], '-')
    #plt.plot_date(tseries, day_can['grass_Ag'], '-')
    plt.plot_date(tseries, day_can['grass_Ag'] + day_can['trees_Ag'], '-')
    plt.show()

    return 1

    plt.plot(-year_blk["GPP"])
    plt.plot(year_can["trees_Ag"])
    plt.plot(year_can["grass_Ag"])
    plt.show()


    mod1 = -exp1_hr['GPP']
    mod2 = exp1_tg['trees_Ag'] + exp1_tg['grass_Ag']

    plt.plot(mod1, mod2, 'o', alpha=0.5)
    plt.show()


if __name__ == "__main__":

    FILEPATH = expanduser("~/Savanna/Data/HowardSprings_IAV/")
    FILENAME1 = FILEPATH + "gasex/spa_treegrass_reduced.pkl"
    FILENAME2 = FILEPATH + "bulkflux/spa_hourly_output.pkl"

    SAVEPATH = expanduser("~/Savanna/Analysis/figures/IAV/HWS_treegrass_gpptrends.pdf")

    main()
