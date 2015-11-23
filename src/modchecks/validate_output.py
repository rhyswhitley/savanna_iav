#!/usr/bin/env python2

import pandas as pd
import cPickle as pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import integrate

def import_spa_output(file_name, start_date, dfreq="30min"):
    """
    Imports dingo meteorology file and puts an index on the time column
    """
    # read in file
    data = pd.read_csv(file_name, sep=r',\s+', engine='python')
    data.columns = [dc.replace(' ', '') for dc in data.columns]
    # re-do the datetime column as it is out by 30 minutes
    data['DATE'] = pd.date_range(start=start_date, periods=len(data), freq=dfreq)
    # set dataframe index on this column
    data_dated = data.set_index(['DATE'])
    # return to user
    return data_dated

def main():
    """Checks whether SPA outputs are matching observations"""

    hws_obs = pickle.load(open(DATFILE, 'rb'))
    spa_out = import_spa_output(SPAFILE, hws_obs.index[0])

    # integrate to daily time-step
    hws_day = hws_obs.ix[:, ["GPP_Con", "Fe_Con"]] \
        .resample('D', how=lambda x: integrate.trapz(x, dx=1800))
    spa_day = spa_out.resample('D', how=lambda x: integrate.trapz(x, dx=1800))


    # plotting

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # LE - smoothed representation
    hws_day_water = pd.rolling_mean(hws_day["Fe_Con"]/1e6, 7, min_periods=1)
    spa_day_water = pd.rolling_mean(spa_day["lemod"]/1e6, 7, min_periods=1)

    ax1.plot_date(hws_day_water.index, hws_day_water, '-', c='darkblue', alpha=0.5)
    ax1.plot_date(spa_day_water.index, spa_day_water, '-', c='blue')

    # GPP
    hws_day_carbon = pd.rolling_mean(hws_day["GPP_Con"]/1e6*12, 7, min_periods=1)
    spa_day_carbon = pd.rolling_mean(spa_day["gpp"]/1e6*12, 7, min_periods=1)

    ax2.plot_date(hws_day_carbon.index, hws_day_carbon, '-', c='darkred', alpha=0.5)
    ax2.plot_date(spa_day_carbon.index, spa_day_carbon, '-', c='red')

    # labels and limits
    ax1.set_ylim([-15, 15])
    ax2.set_ylim([-15, 15])

    ax1.set_title("Howard Springs H$_{2}$O x CO$_{2}$ (2001 to 2015)", fontsize=14)
    ax1.set_ylabel(r'LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=14)
    ax2.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=14)

    plt.grid(c='gray')
    plt.subplots_adjust(left=0.1, right=0.9)
    plt.savefig(SAVEPATH)


if __name__ == "__main__":

    DATPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/")
    DATFILE = DATPATH + "tower_fluxes_valid.pkl"
    SAVEPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/validated_exp1.pdf")

    FILEPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SPAFILE = FILEPATH + "HS_Exp1/outputs/hourly.csv"

    main()

