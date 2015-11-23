#!/usr/bin/env python2

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from scipy import integrate

def create_phen_file(lai_ts):
    total = lai_ts.resample('D', how='mean')
    lai_veg = treegrass_frac(total, 30)
    site_phen = divide_leaves(lai_veg)

    site_phen['rootbiomass'] = [3840.]*len(total)
    site_phen['nitrogen'] = [lai*2 for lai in total]
    site_phen['TimeStep'] = total.index.dayofyear

    site_phen_rot = site_phen.iloc[:, [13, 11] + range(0, 11) + [12]]
    site_phen_out = pd.concat([site_phen_rot, site_phen.iloc[:, 1:11]], axis=1)
    site_phen_out.columns = ['DOY', 'rootbiom', 'lai'] \
        + ['lfr_{0}'.format(i) for i in range(1, 11)] \
        + ['nit'] \
        + ['nfr_{0}'.format(i) for i in range(1, 11)]
    return site_phen_out

def divide_leaves(lai_part):
    trees_frac = [lai_part['tree']*0.7/lai_part['total']/5. for i in range(5)]
    grass_frac = [lai_part['grass']*1.3/lai_part['total']/5. for i in range(5)]
    leaf_alloc = pd.concat([lai_part['total']] + trees_frac + grass_frac, axis=1)
    return leaf_alloc

def treegrass_frac(ndvi, day_rs):
    """
    Process based on Donohue et al. (2009) to separate out tree and grass cover,
    using moving windows (adapted here for daily time-step)
    """
    # first calculate the 7-month moving minimum window across the time-series
    # changed period to 3 to kill grass in dry season
    fp1 = moving_something(np.min, ndvi, period=3, day_rs=day_rs)
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

def import_tower_data(file_name):
    """
    Imports dingo meteorology file and puts an index on the time column
    """
    # read in file
    tower = pd.read_csv(file_name)
    # re-do the datetime column as it is out by 30 minutes
    tower['DATE'] = pd.date_range(start="2001-01-01", periods=len(tower), freq="30min")
    # set dataframe index on this column
    tower_dated = tower.set_index(['DATE'])
    # return to user
    return tower_dated

def clean_tower_data(dataset):
    """
    Cleans the dataset for misssing values, using a linear interpolation and backfill on
    missing values.

    Positive, non-physical values are removed by comparing values with the max of a 95% CI
    monthly moving window.

    Negative, non-physical values are set to 0
    """
    # interpolate and back fill missing values
    data_fill = dataset.interpolate().fillna(method='bfill')

    # pick columns to clean
    met_data = data_fill.ix[:, ["Ta_Con", "Fsd_Con", "VPD_Con"]]

    # remove non-physical values (assume all are positive)
    data_clean = met_data.apply(remove_nonphysical)
    # add PAR data
    data_clean["PAR"] = data_fill["Fsd_Con"]*2.3
    # add timestep column for light interception geometric calculations
    data_clean["TimeStep"] = [d.dayofyear + (d.hour + d.minute/60.)/24 \
                              for d in data_fill.index]

    # add extra uncleaned columns
    data_out = pd.concat([data_clean, \
                         data_fill.ix[:, ["CO2", "Ws_CSAT_Con", "Precip_Con"]]], \
                         axis=1)
    # return data in this order of columns
    return data_out.ix[:, ["TimeStep", "Ta_Con", "CO2", "Ws_CSAT_Con", "Fsd_Con", "VPD_Con", "PAR", "Precip_Con"]]

def remove_nonphysical(dstream, win=30):
    """
    Non-physical values are removed by comparing values with the max of a 95% CI
    monthly moving window.

    Negative values are set to 0
    """
    # rolling mean
    mean = pd.rolling_mean(dstream, window=win*48).fillna(method="bfill")
    # rolling standard deviation
    std = pd.rolling_std(dstream, window=win*48).fillna(method="bfill")
    # determined rolling ci
    ci99 = [m + 2.5*s for (m, s) in zip(mean, std)]
    # max CI 99
    top_val = np.max(ci99)
    # clean values
    #dstream_clean = [np.min([top_val, ds[i]]) for (i, ds) in enumerate(dstream)]
    dstream_clean = np.minimum(top_val, np.maximum(dstream, 0))
    # return cleaned data stream
    return dstream_clean

def expand_climatology(dataset):
    """
    Takes the one-year (366 day) climatology and builds a 14 year dataset from it
    """
    # remove leap day
    non_leap = dataset[~((dataset.index.day == 29) & (dataset.index.month == 2))]
    # build new time-series using non-leap years + leap year
    grp_year = pd.concat([non_leap]*3 + [dataset], axis=0)
    # there are 3 groups of non-leap + leap years PLUS two normal years
    full_tseries = pd.concat([grp_year]*3 + [non_leap]*2, axis=0)
    # returns a repeating climatological time-series over 14 years
    full_tseries['DT'] = pd.date_range(start="2001-01-01", periods=len(full_tseries), freq="30 min")
    full_tseries2 = full_tseries.reset_index(drop=True)
    return full_tseries2.set_index(['DT'])

def plot_inputs(dataset, phen, EX=1):
    n_plots = 6
    gs = gridspec.GridSpec(n_plots, 1)
    ax1 = [plt.subplot(gs[i]) for i in range(n_plots)]
    # turn off x labels
    for (i, subax) in enumerate(ax1):
        if i < len(ax1) - 1:
            subax.axes.get_xaxis().set_visible(False)

    # plots
    time_x = pd.date_range(start="2001-01-01", end="2014-12-31", freq='D') #.to_pydatetime()
    ax1[0].plot_date(time_x, dataset["Fsd_Con"].resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6), '-', c='red')
    ax1[1].plot_date(time_x, dataset["VPD_Con"].resample('D', how='mean'), '-', c='purple')
    ax1[2].plot_date(time_x, dataset["Ta_Con"].resample('D', how='mean'), '-', c='orange')
    ax1[3].plot_date(time_x, dataset["Ws_CSAT_Con"].resample('D', how='mean'), '-', c='darkgreen')
    ax1[4].plot_date(time_x, dataset["Precip_Con"].resample('D', how='sum'), '-', c='blue')
    ax2 = ax1[4].twinx()
    ax2.plot_date(time_x, dataset["CO2"].resample('D', how='mean'), '-', c='black', lw=2)
    ax1[5].plot_date(time_x, phen["lai"], '-', c='green')

    # labels
    plt_label = "Howard Springs Experiment {0} Inputs".format(EX)
    ax1[0].set_title(plt_label)
    ax1[0].set_ylabel("$R_{s}$ (MJ m$^{-2}$)")
    ax1[1].set_ylabel("$D_{v}$ (kPa)")
    ax1[2].set_ylabel("$T_{a}$ ($\degree$C)")
    ax1[3].set_ylabel("$U_{v}$ (m s$^{-1}$)")
    ax1[4].set_ylabel("$PPT$ (mm)")
    ax1[5].set_ylabel("LAI")
    ax2.set_ylabel("CO$_{2}$ (ppm)")
    for i in range(len(ax1)):
        ax1[i].yaxis.set_label_coords(-0.07, 0.5)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.1)
    plt.savefig(figure_path + plt_label.replace(' ', '_') + ".pdf", rasterized=True)
    return None

def mix_climatology(dataset1, dataset2, ycol):
    dataset3 = dataset1.copy()
    dataset3[ycol] = dataset2[ycol]
    return dataset3

def main():

    # import datasets to create experiment input files
    climo_raw = import_one_year(clim_met_file)
    tower_raw = import_tower_data(ec_tower_file)

    # expand climatology out to 14 years (2001 to 2015)
    climo_14yr = expand_climatology(climo_raw)

    # universal phenology file
    spa_phen_1 = create_phen_file(tower_raw["Lai_1km_new_smooth"])
    # experiment 2
    spa_phen_2 = create_phen_file(climo_14yr["Lai_1km_new_smooth"])

    # universal phenology file
    spa_phen_1.to_csv(input_phen_1, sep=",", index=False)
    spa_phen_2.to_csv(input_phen_2, sep=",", index=False)

    # experiment 1
    spa_met_1 = clean_tower_data(tower_raw)
    # experiment 2
    spa_met_2 = clean_tower_data(climo_14yr)

    # save these figures to make sure input files are constructed correctly
    plot_inputs(spa_met_1, spa_phen_1, 1)
    plot_inputs(spa_met_2, spa_phen_2, 2)

    # experiment simulations
    spa_met_1.to_csv(input_folders[0], sep=",", index=False)
    spa_met_2.to_csv(input_folders[1], sep=",", index=False)

    # swap on these variables
    var_on = ["CO2", "Ta_Con", "Precip_Con", "Fsd_Con", "VPD_Con"]

    # experiment 3 to 7
    spa_met_x = [mix_climatology(spa_met_2, spa_met_1, vo) for vo in var_on]
    for i in range(len(var_on)):
        spa_met_x[i].to_csv(input_folders[2 + i], sep=",", index=False)
        exp_int = i + 3
        plot_inputs(spa_met_x[i], spa_phen_2, exp_int)

    # WRITE TO CSV FILES
    return 1




if __name__ == "__main__":

    clim_met_file = expanduser("~/Dropbox/30 minute met driver climatology v12a HowardSprings.csv")
    ec_tower_file = expanduser("~/Dropbox/30 minute met driver 2001-2015 v12a HowardSprings.csv")

    input_path = expanduser("~/Savanna/Models/SPA1/outputs/site_co2/HowardSprings/inputs/")
    input_folders = ["{0}hs_met_exp_{1}.csv".format(input_path, i) for i in range(1, 8)]
    input_phen_1 = input_path + "hs_phen_exp_1.csv"
    input_phen_2 = input_path + "hs_phen_exp_all.csv"

    figure_path = expanduser("~/Savanna/Analysis/figures/IAV/inputs/")

    main()
