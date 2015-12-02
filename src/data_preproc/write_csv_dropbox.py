#!/usr/bin/env python

import os
import re
import netCDF4 as nc
import pandas as pd
from natsort import natsorted
# mpi
from joblib import delayed, Parallel
from multiprocessing import cpu_count

def get_value(nc_obj, label):
    return nc_obj.variables[label][:].flatten()

def get_dataframe(nc_path):
    """
    A quick function to transform a netcdf file into a pandas dataframe that
    can be used for analysis and plotting. Attributes are extracted using
    in built netCDF4 library functions. Time is arbitrary and needs to be
    set by the user.
    """
    print("> extracting contents in object at {0}".format(nc_path))
    # make a connection to the netCDF file
    ncdf_con = nc.Dataset(nc_path, 'r', format="NETCDF4")
    # number of rows, equivalent to time-steps
    time_len = len(ncdf_con.dimensions['time'])
    # extract time information
    time_sec = ncdf_con.variables['time']
    sec_orig = re.search(r'\d+.*', str(time_sec.units)).group(0)
    # the header values for each measurements; excludes time and space components
    nc_allkeys = ncdf_con.variables.keys()
    # only want tree and grass outputs
    data_values = [key for key in nc_allkeys \
                   if re.search('(GPP)|(AutoResp)|(Qle)|(Esoil)|(Tveg)|(Ecanop)}', key)]

    # create a new dataframe from the netCDF file
    nc_dataframe = pd.DataFrame({label: get_value(ncdf_con, label) \
                                for label in data_values}, \
                                index=pd.date_range(sec_orig, \
                                periods=time_len, freq="30min"))
    return nc_dataframe

def main():

    # Get the number of available cores for multi-proc
    num_cores = cpu_count()

    # Get the filepaths for each experiment's output ncdf file
    nc_paths = natsorted([os.path.join(dp, f) for (dp, dn, fn) in os.walk(DIRPATH) \
                    for f in fn if re.search("^((?!DS_Store|inputs).)*$", f)])

    # Retrieve dataframes of tree and grass productivity from ncdf files
    hws_dfs = Parallel(n_jobs=num_cores)(delayed(get_dataframe)(npf) \
                    for npf in nc_paths)

    # write to csv for Lindsay and Jason
    for (i, df) in enumerate(hws_dfs):
        fname = "Exp_{0}_HourlyFlux.csv".format(i+1)
        print("Writing file: {0}".format(fname))
        df.to_csv(CSVPATH + fname, spe=",", index=True, float_format='%.3f')

    return None

if __name__ == '__main__':

    DIRPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/")
    CSVPATH = os.path.expanduser("~/Dropbox/HWS Long term paper/SPA_Simulations/Hourly_Outputs/")

    main()
