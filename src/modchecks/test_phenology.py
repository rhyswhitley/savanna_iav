#!/usr/bin/env python2

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
import scipy

def treegrass_frac(ndvi, day_rs):
    """
    Process based on Donohue et al. (2009) to separate out tree and grass cover,
    using moving windows (adapted here for daily time-step)
    """
    # first calculate the 7-month moving minimum window across the time-series
    # period = 7
    fp1 = moving_something(np.min, ndvi, period=3, day_rs=day_rs)
    # period = 9
    fp2 = moving_something(lambda x: sum(x)/(9*day_rs), fp1, period=9, day_rs=day_rs)
    fr1 = ndvi - fp2

    ftree = [p2 - np.abs(r1) if r1 < 0 else p2 for p2, r1 in zip(fp2, fr1)]
    fgrass = ndvi - ftree

    return pd.DataFrame({'total':ndvi, 'tree':ftree, 'grass':fgrass})

def moving_something(_fun, tseries, period, day_rs=16, is_days=True):
    """
    Applies a function to a moving window of the time-series:
    ft_ = function([ f(t-N), f(t). f(t+N)])
    """
    # if the time-series is at a day-time step, update the window to a step-size of 16 days
    if is_days:
        p0 = period*day_rs
    else:
        p0 = period

    # find upper and lower bounds of the moving window
    half = p0//2
    tlen = len(tseries)

    twin = [0]*tlen
    for im in range(tlen):
        # find the something for the window that satisfy the edge conditions
        if im < half:
            # fold back onto the end of the time-series
            twin[im] = _fun(np.hstack([tseries[tlen-(half-im):tlen],\
                                        tseries[0:im+half]]))
        elif im > tlen-half:
            # fold back into the beginning of the time-series
            twin[im] = _fun(np.hstack([tseries[im-half:tlen],\
                                        tseries[0:half-(tlen-im)]]))
        else:
            twin[im] = _fun(tseries[im-half:im+half])

    return twin

def import_one_year(file_name):
    """
    Imports the one-year climatology, resetting time columns
    as a multi-index pandas dataframe
    """
    # universal time labels
    time_label = ['Month', 'Day', 'Hour', 'Min']
    # import data
    clim_raw = pd.read_csv(clim_met_file)
    # fix column names
    clim_raw.columns = time_label + list(clim_raw.columns[4:])
    # datetime column
    clim_raw['DT'] = pd.date_range("2004-01-01", periods=len(clim_raw), freq="30min")
    # index on time
    clim_data = clim_raw.set_index('DT')
    # return to user
    return clim_data.ix[:, 4:]

def main():

    climo_raw = import_one_year(clim_met_file).resample('D', how='mean')

    phen = treegrass_frac(climo_raw["Lai_1km_new_smooth"], 30)

    plt.plot(phen.tree*0.7, c='blue')
    plt.plot(phen.grass*1.3, c='red')
    plt.show()

    return None

if __name__ == "__main__":

    clim_met_file = expanduser("~/Dropbox/30 minute met driver climatology v12a HowardSprings.csv")
    ec_tower_file = expanduser("~/Dropbox/30 minute met driver 2001-2015 v12a HowardSprings.csv")

    main()
