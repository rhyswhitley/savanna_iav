#!/usr/bin/env python2

import os, errno
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style
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

def mix_climatology(dataset1, dataset2, ycol):
    dataset3 = dataset1.copy()
    dataset3[ycol] = dataset2[ycol]
    return dataset3

def smooth_data(dataset):
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    # sample functions
    samplers = [np.mean]*4 + [dayint, np.mean, dayint, np.sum]
    # dictionary to pass to resample
    sample_dict = {lab: samp for (lab, samp) in zip(dataset.columns, samplers*2)}

    # downsampled to daily time-scale
    daily_data = dataset.resample("D", how=sample_dict)

    # smooth data using a 14-day rolling mean
    rolling = lambda x: pd.rolling_mean(x, window=14, min_periods=1)
    new_dataset = daily_data.apply(rolling, axis=0)

    # return new dataset
    return new_dataset

def plot_inputs(dataset, phen, swap_var=None, EX=1):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = plt.rcParams['axes.color_cycle']
    n_plots = 6

    #import ipdb; ipdb.set_trace()
    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(n_plots, 1)
    ax1 = [plt.subplot(gs[i]) for i in range(n_plots)]

     # turn off x labels
    for i in range(5):
        ax1[i].xaxis.set_ticklabels([])

    agg_data = smooth_data(dataset)
    agg_data['lai'] = phen['lai']

    # plots
    date_ticks = pd.date_range("2001", periods=15, freq='AS')
    time_x = agg_data.index

    plot_vars = ["Fsd_Con", "VPD_Con", "Ta_Con", "Ws_CSAT_Con", "Precip_Con", "lai"]
    pcols = [ncols[0]]*7
    if swap_var is not None:
        if swap_var == "CO2":
            swap_ix = 4;
        else:
            swap_ix = plot_vars.index(swap_var)
        pcols[swap_ix] = ncols[1]

    for (i, pvar) in enumerate(plot_vars):
        if i != 4:
            ax1[i].plot_date(time_x, agg_data[pvar], '-', c=pcols[i])
        else:
            if swap_var == "CO2":
                bcol = ncols[0]
                ccol = pcols[i]
            elif swap_var == "Precip_Con":
                bcol = pcols[i]
                ccol = ncols[0]
            else:
                bcol = pcols[i]
                ccol = pcols[i]
            ax1[i].bar(time_x, agg_data[pvar], color=None, edgecolor=bcol, alpha=0.3)
            ax1[i].xaxis_date()
            ax2 = ax1[i].twinx()
            ax2.plot_date(time_x, agg_data["CO2"], '-', c=ccol, lw=2.5)
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

    # custom legend lines
    held_line = mlines.Line2D([], [], color=ncols[0], lw=2, marker=None, label="held variables")
    pert_line = mlines.Line2D([], [], color=ncols[1], lw=2, marker=None, label="perturbed variable")

    # plot legend
    ax1[5].legend(handles=[held_line, pert_line], bbox_to_anchor=(0.5, -0.4), \
                  loc='upper center', ncol=2, prop={'size':10})

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.15)

    # saving
    figure_path = os.path.expanduser("~/Savanna/Analysis/figures/IAV/inputs/")
    plt.savefig(figure_path + plt_label.replace(' ', '_') + ".pdf", rasterized=True)

    return None

def assign_variables(nc_obj):
    # CREATE DIMENSIONS
    nc_obj.createDimension('x', 1)
    nc_obj.createDimension('y', 1)
    nc_obj.createDimension('z', 1)
    nc_obj.createDimension('time', None)
    # CREATE VARIABLES
    nc_obj.createVariable('x', 'f8', ('x'))
    nc_obj.createVariable('y', 'f8', ('y'))
    nc_obj.createVariable('latitude', 'f8', ('x', 'y'))
    nc_obj.createVariable('longitude', 'f8', ('x', 'y'))
    nc_obj.createVariable('time', 'f8', ('time'))
    # >> [Time-varying values]
    # >> Local Meteorology
    nc_obj.createVariable('SWdown', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Tair', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('VPD', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Cair', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Wind', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Rainfall', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('LAI', 'f8', ('time', 'x', 'y'))
    # >> Climatologies
    nc_obj.createVariable('clim_SWdown', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_Tair', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_VPD', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_Cair', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_Wind', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_Rainfall', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('clim_LAI', 'f8', ('time', 'x', 'y'))
    return None

def assign_units(nc_obj, start_date):
    # ASSIGN UNITS
    # >> [Dimensions]
    nc_obj.variables['x'].units = ""
    nc_obj.variables['y'].units = ""
    nc_obj.variables['latitude'].units = "degrees_north"
    nc_obj.variables['longitude'].units = "degrees_east"
    nc_obj.variables['time'].units = "seconds since " + start_date
    # >> [Time-varying values]
    # >> Local Meteorology
    nc_obj.variables['SWdown'].units = "W/m^2"
    nc_obj.variables['Tair'].units = "degrees Celsius"
    nc_obj.variables['VPD'].units = "kPa"
    nc_obj.variables['Cair'].units = "umol/mol"
    nc_obj.variables['Wind'].units = "m/s"
    nc_obj.variables['Rainfall'].units = "mm"
    nc_obj.variables['LAI'].units = "m^2/m^2"
    # >> Climatologies
    nc_obj.variables['clim_SWdown'].units = "W/m^2"
    nc_obj.variables['clim_Tair'].units = "degrees Celsius"
    nc_obj.variables['clim_VPD'].units = "kPa"
    nc_obj.variables['clim_Cair'].units = "umol/mol"
    nc_obj.variables['clim_Wind'].units = "m/s"
    nc_obj.variables['clim_Rainfall'].units = "mm"
    nc_obj.variables['clim_LAI'].units = "m^2/m^2"
    return None

def assign_longNames(nc_obj):
    # LONG NAMES
    nc_obj.variables['SWdown'].longname = "Downwelling shortwave radiation"
    nc_obj.variables['Tair'].longname = "Air temperature"
    nc_obj.variables['VPD'].longname = "Vapour pressure deficit"
    nc_obj.variables['Cair'].longname = "Atmospheric CO2 concentration"
    nc_obj.variables['Wind'].longname = "Wind speed"
    nc_obj.variables['Rainfall'].longname = "Precipitation"
    nc_obj.variables['LAI'].longname = "MODIS 8-day composite leaf area index"
    # Vegetation
    nc_obj.variables['clim_SWdown'].longname = "Downwelling shortwave radiation"
    nc_obj.variables['clim_Tair'].longname = "Air temperature"
    nc_obj.variables['clim_VPD'].longname = "Vapour pressure deficit"
    nc_obj.variables['clim_Cair'].longname = "Atmospheric CO2 concentration"
    nc_obj.variables['clim_Wind'].longname = "Wind speed"
    nc_obj.variables['clim_Rainfall'].longname = "Precipitation"
    nc_obj.variables['clim_LAI'].longname = "MODIS 8-day composite leaf area index"
    return None

def ensure_dir(path):
    # Create folders for storage if they done exist
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def main():

    #----------------------------------------------------------------------
    # STAGE DATA
    #----------------------------------------------------------------------

    # create the directories if they don't exist
    [ensure_dir(fpath) for fpath in INPUT_FOLDERS]

    # import datasets to create experiment input files
    climo_raw = import_one_year(clim_met_file)
    tower_raw = import_tower_data(ec_tower_file)

    # expand climatology out to 14 years (2001 to 2015)
    climo_14yr = expand_climatology(climo_raw)


    #----------------------------------------------------------------------
    # METEOROLOGY FILE CREATION
    #----------------------------------------------------------------------

    # experiment 1
    spa_met_1 = clean_tower_data(tower_raw)
    # experiment 2
    spa_met_2 = clean_tower_data(climo_14yr)

    # swap on these variables
    var_on = ["CO2", "Ta_Con", "Precip_Con", "Fsd_Con", "VPD_Con"]
    # experiment 3 to 7
    spa_met_x1 = [mix_climatology(spa_met_2, spa_met_1, vo) for vo in var_on]
    # experiments 9 to 14
    spa_met_x2 = [mix_climatology(spa_met_1, spa_met_2, vo) for vo in var_on]

    # write experiment simulations to file
    spa_met_1.to_csv(INPUT_FILES1[0], sep=",", index=False)
    spa_met_2.to_csv(INPUT_FILES1[1], sep=",", index=False)
    for (ix, (spa_df1, spa_df2)) in enumerate(zip(spa_met_x1, spa_met_x2)):
        spa_df1.to_csv(INPUT_FILES1[2 + ix], sep=",", index=False)
        spa_df2.to_csv(INPUT_FILES2[ix], sep=",", index=False)


    #----------------------------------------------------------------------
    # PHENOLOGY FILE CREATION
    #----------------------------------------------------------------------

    # universal phenology file
    spa_phen_1 = create_phen_file(tower_raw["Lai_1km_new_smooth"])
    # experiment 2
    spa_phen_2 = create_phen_file(climo_14yr["Lai_1km_new_smooth"])

    # universal phenology file
    spa_phen_1.to_csv(INPUT_PHEN_1, sep=",", index=False)
    spa_phen_2.to_csv(INPUT_PHEN_2, sep=",", index=False)


    #----------------------------------------------------------------------
    # CREATE SYMBOLIC LINKS
    #----------------------------------------------------------------------


    #----------------------------------------------------------------------
    # PLOT OUTPUTS **CHECKING
    #----------------------------------------------------------------------

    # experiments 1 and 2
    plot_inputs(spa_met_1, spa_phen_1, None, 1)
    plot_inputs(spa_met_2, spa_phen_2, None, 2)

    # experiments 8 and 14
    plot_inputs(spa_met_2, spa_phen_1, "lai", 8)
    plot_inputs(spa_met_1, spa_phen_2, "lai", 14)

    # experiments 3 to 7 & 9 to 14
    for (ic, (spa_df1, spa_df2)) in enumerate(zip(spa_met_x1, spa_met_x2)):
        plot_inputs(spa_df1, spa_phen_2, var_on[ic], ic + 3)
        plot_inputs(spa_df2, spa_phen_1, var_on[ic], ic + 9)


    #----------------------------------------------------------------------
    # NETCDF CREATION
    #----------------------------------------------------------------------

    # up-sample LAI to the same timestep as meteorology for ncdf export
    LAI_phen1_30min = spa_phen_1["lai"].resample('30min', fill_method="ffill")
    LAI_phen2_30min = spa_phen_2["lai"].resample('30min', fill_method="ffill")

    # Open a NCDF4 file for SPA simulation outputs
    nc_fout = NCSAVEPATH + "spa_hws_inputs.nc"
    nc_file = nc.Dataset(nc_fout, 'w', format='NETCDF4')

    assign_variables(nc_file)
    assign_units(nc_file, "2001-01-01 00:00:30")
    assign_longNames(nc_file)

    # Assign values to variables
    tseries = pd.timedelta_range(0, periods=len(spa_met_1), freq="1800s") \
                .astype('timedelta64[s]')

    # Get time from netcdf driver file
    nc_file.variables['time'][:] = np.array(tseries)
    # Local Meteorologies
    nc_file.variables['SWdown'][:] = np.array(spa_met_1['Fsd_Con'])
    nc_file.variables['VPD'][:] = np.array(spa_met_1['VPD_Con'])
    nc_file.variables['Tair'][:] = np.array(spa_met_1['Ta_Con'])
    nc_file.variables['Cair'][:] = np.array(spa_met_1['CO2'])
    nc_file.variables['Wind'][:] = np.array(spa_met_1['Ws_CSAT_Con'])
    nc_file.variables['Rainfall'][:] = np.array(spa_met_1['Precip_Con'])
    nc_file.variables['LAI'][:] = np.array(LAI_phen1_30min)
    # Climatologies
    nc_file.variables['clim_SWdown'][:] = np.array(spa_met_2['Fsd_Con'])
    nc_file.variables['clim_VPD'][:] = np.array(spa_met_2['VPD_Con'])
    nc_file.variables['clim_Tair'][:] = np.array(spa_met_2['Ta_Con'])
    nc_file.variables['clim_Cair'][:] = np.array(spa_met_2['CO2'])
    nc_file.variables['clim_Wind'][:] = np.array(spa_met_2['Ws_CSAT_Con'])
    nc_file.variables['clim_Rainfall'][:] = np.array(spa_met_2['Precip_Con'])
    nc_file.variables['clim_LAI'][:] = np.array(LAI_phen2_30min)

    nc_file.close()

    return 1

if __name__ == "__main__":

    clim_met_file = os.path.expanduser("~/Dropbox/30 minute met driver climatology v12a HowardSprings.csv")
    ec_tower_file = os.path.expanduser("~/Dropbox/30 minute met driver 2001-2015 v12a HowardSprings.csv")

    INPUT_PATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2")
    INPUT_FOLDERS = ["{0}/HS_Exp{1}/inputs".format(INPUT_PATH, i) for i in range(1, 8) + range(9, 14)]

    INPUT_FILES1 = ["{0}/hs_met_exp_{1}.csv".format(path, i+1) \
                    for (i, path) in enumerate(INPUT_FOLDERS[:7])]
    INPUT_FILES2 = ["{0}/hs_met_exp_{1}.csv".format(path, i+9) \
                    for (i, path) in enumerate(INPUT_FOLDERS[7:])]

    INPUT_PHEN_1 = "{0}/common_inputs/hs_phen_exp_1.csv".format(INPUT_PATH)
    INPUT_PHEN_2 = "{0}/common_inputs/hs_phen_exp_all.csv".format(INPUT_PATH)

    NCSAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/")

    main()

