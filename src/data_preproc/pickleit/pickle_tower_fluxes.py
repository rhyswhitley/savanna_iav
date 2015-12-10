#!/usr/bin/env python

import os
import re
import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle

def get_value(nc_obj, label):
    return nc_obj.variables[label][:].flatten()

def get_dataframe(nc_path):
    """
    A quick function to transform a netcdf file into a pandas dataframe that
    can be used for analysis and plotting. Attributes are extracted using
    in built netCDF4 library functions. Time is arbitrary and needs to be
    set by the user.
    """
    print("> pickling contents in object at {0}".format(nc_path))
    # make a connection to the netCDF file
    ncdf_con = nc.Dataset(nc_path, 'r', format="NETCDF4")
    # number of rows, equivalent to time-steps
    time_len = len(ncdf_con.dimensions['time'])
    # extract time information
    time_sec = ncdf_con.variables['time']
    sec_orig = re.search(r'\d+.*', str(time_sec.units)).group(0)
    # the header values for each measurements; excludes time and space components
    nc_allkeys = ncdf_con.variables.keys()
    # only want time-varying inputs
    data_values = [key for key in nc_allkeys \
                   if re.search("^((?!x|y|time|latitude|longitude).)*$", key)]

    # create a new dataframe from the netCDF file
    nc_dataframe = pd.DataFrame({label: get_value(ncdf_con, label) \
                                for label in data_values}, \
                                index=pd.date_range(sec_orig, \
                                periods=time_len, freq="30min"))
    return nc_dataframe

def main():

    # Retrieve dataframes of tree and grass productivity from ncdf files
    input_df = get_dataframe(DIRPATH)

    # pickle the leaf scale outputs (see if it's quicker to load)
    pickle.dump(input_df, open(PKLPATH+"hourly/tower_fluxes_valid.pkl", "wb"))

    return None

if __name__ == '__main__':

    DIRPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/hws_tower_fluxes.nc")
    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
