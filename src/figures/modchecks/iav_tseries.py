#!/usr/bin/env python2

import collections
import matplotlib.pyplot as plt
import pandas as pd
import os, re
import numpy as np
from pylab import get_cmap

def import_spa_output(file_name, start_date="2001-01-01", dfreq="30min"):
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

    filepaths = [os.path.join(dp, f) \
                 for (dp, dn, fn) in os.walk(FILEPATH) \
                 for f in fn if re.search(r"daily", f)]

    dailyflux0 = {'exp_{0}'.format(i+1): \
                import_spa_output(fp, dfreq='D') \
                for (i, fp) in enumerate(filepaths)} #\

    dailyflux = collections.OrderedDict(sorted(dailyflux0.iteritems()))


    # plotting below here
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    col_h2o = get_cmap("winter")(np.linspace(0, 1, len(dailyflux)))
    col_co2 = get_cmap("hot")(np.linspace(0, 1, len(dailyflux)))

    for (i, (exp_lab, exp_dat)) in enumerate(dailyflux.iteritems()):

        exp_dat.fillna(method="pad", inplace=True)
        # smooth predictions
        spa_water = pd.rolling_mean(exp_dat["et"], 7, min_periods=1)
        spa_carbon = pd.rolling_mean(exp_dat["gpp"], 7, min_periods=1)

        # LE
        ax1.plot_date(spa_water.index, spa_water, '-', \
                      c=col_h2o[i], alpha=0.7, label=exp_lab)
        # GPP
        ax2.plot_date(spa_carbon.index, -spa_carbon, '-', \
                      c=col_co2[i], alpha=0.7, label=exp_lab)

    # labels and limits
    ax1.set_ylim([-8, 8])
    ax2.set_ylim([-8, 8])

    ax1.set_title("Howard Springs H$_{2}$O x CO$_{2}$ (2001 to 2015)", fontsize=14)
    ax1.set_ylabel(r'ET (mm m$^{-2}$ d$^{-1}$)', fontsize=14)
    ax2.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=14)

    plt.grid(c='gray')
    plt.subplots_adjust(left=0.1, right=0.9)
    plt.savefig(SAVEPATH + "exp_flux_tseries.pdf")


    return 1

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SAVEPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/")

    main()
