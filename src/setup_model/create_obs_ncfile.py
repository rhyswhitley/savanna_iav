#!/usr/bin/env python2

import os, re
import numpy as np
import pandas as pd
import netCDF4 as nc

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
    nc_obj.createVariable('NEE', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('GPP', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Resp', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Qle', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Qh', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('Qn', 'f8', ('time', 'x', 'y'))
    nc_obj.createVariable('SoilMoist10', 'f8', ('time', 'x', 'y'))

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
    nc_obj.variables['NEE'].units = "umol/m^2/s^1"
    nc_obj.variables['GPP'].units = "umol/m^2/s^1"
    nc_obj.variables['Resp'].units = "umol/m^2/s^1"
    nc_obj.variables['Qle'].units = "W/m^2"
    nc_obj.variables['Qh'].units = "W/m^2"
    nc_obj.variables['Qn'].units = "W/m^2"
    nc_obj.variables['SoilMoist10'].units = "m^3/m^3"
    return None

def assign_longNames(nc_obj):
    # LONG NAMES
    nc_obj.variables['NEE'].longname = "Net ecosystem exchange"
    nc_obj.variables['GPP'].longname = "Gross primary productivity"
    nc_obj.variables['Resp'].longname = "Autotropihc respiration"
    nc_obj.variables['Qle'].longname = "Latent heat"
    nc_obj.variables['Qh'].longname = "Sensible heat"
    nc_obj.variables['Qn'].longname = "Net radiation"
    nc_obj.variables['SoilMoist10'].longname = "Soil moisture content at 10 cm depth"


def main():

    tower_data = pd.read_csv(DIRPATH, index_col='DT', parse_dates=True)

    get_cols = ["Fe_Con", "Fc_Con", "Fc_ustar", "Fn_Con", "Fh_Con", \
                "Sws_Con", "GPP_Con", "Fre_Con"]

    subset_tower = tower_data.ix[:, get_cols]

    # Open a NCDF4 file for SPA simulation outputs
    nc_file = nc.Dataset(NCSAVEPATH, 'w', format='NETCDF4')

    assign_variables(nc_file)
    assign_units(nc_file, "2001-01-01 00:00:30")
    assign_longNames(nc_file)

    # Assign values to variables
    tseries = pd.timedelta_range(0, periods=len(subset_tower), freq="1800s") \
                .astype('timedelta64[s]')

    # Get time from netcdf driver file
    nc_file.variables['time'][:] = np.array(tseries)
    # Local Meteorologies
    nc_file.variables['NEE'][:] = subset_tower['Fc_Con'].values
    nc_file.variables['GPP'][:] = subset_tower['GPP_Con'].values
    nc_file.variables['Resp'][:] = subset_tower['Fre_Con'].values
    nc_file.variables['Qle'][:] = subset_tower['Fe_Con'].values
    nc_file.variables['Qh'][:] = subset_tower['Fh_Con'].values
    nc_file.variables['Qn'][:] = subset_tower['Fn_Con'].values
    nc_file.variables['SoilMoist10'][:] = subset_tower['Sws_Con'].values

    nc_file.close()

    return 1

if __name__ == "__main__":

    NCSAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/hws_tower_fluxes.nc")
    DIRPATH = os.path.expanduser("~/Dropbox/Dingo12/Advanced_processed_data_HowardSprings_v12.csv")

    main()

