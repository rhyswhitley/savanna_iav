import numpy as np
import natsort
import pandas as pd
import sys, os, re
import time
import netCDF4 as nc
import spa_ncdf_class as spa_nc

__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = "2015-10-15"
__modified__ = time.strftime("%c")
__version__ = '1.0'
__status__ = 'prototype'

class SPAoutput_ETL(object):
    """
    A set of functions that perform pre-processing routines on the raw SPA v1
    output CSV files. These files are imported into pandas dataframes at a
    30min or 1hr time-step. Each output file is saved as a dataframe and then
    pickled for later use.

        dirpath = directory where spa outputs are located
        savepath = directory where pickle and CSVs will be saved

    The file processed as follows:
    1. hourly.csv
    2. l1-10.csv
    3. soilwater.csv
    4. upfrac.csv
    """
    def __init__(self):
        super(SPAoutput_ETL, self).__init__()
        self.start_date = "2001-01-01"
        self.time_freq = "30min"
        self.soil_depth = None
        self.glay = 6 # index in canopy layers where grasses begin

    def reduce_canopy(self, data_list):
        """
        Generalise the 10 canopy layer files into a dictionary of tree and
        grass dataframes
        """
        # split layer files into trees and grasses and
        # concatenate these together
        trees_raw = pd.concat(data_list[:self.glay], axis=0)
        grass_raw = pd.concat(data_list[self.glay:], axis=0)

        # columns to be summed
        sum_col = [1, 2, 6, 8, 9, 13]
        # create dictionary to apply fun to each column via label
        met_labs = {ml: "sum" if i in sum_col else "mean" \
                    for (i, ml) in enumerate(trees_raw.columns)}

        # group on each time-step and aggregate across canopy layers
        trees_grp = trees_raw.groupby(level=0).aggregate(met_labs)
        grass_grp = grass_raw.groupby(level=0).aggregate(met_labs)

        return {'trees' : trees_grp, 'grass' : grass_grp}

    def flatten_dictlist(self, dlist):
        """Flattens the tree/grass dictionary into a single dataframe"""
        # concatenate the two dataframes
        tg_df = pd.concat(dlist, axis=1)
        # set new column names
        tg_df.columns = ["{0}_{1}".format(h1, h2) for (h1, h2) in tg_df.columns]
        # return to user
        return tg_df

    def import_spa_canopy(self, file_name, clayer):
        """
        Imports SPA canopy layer outputs into a pandas dataframe
        * adds indexes on time and canopy layer
        """
        # read in file
        data = pd.read_csv(file_name, sep=r',\s+', engine='python')
        # remove header whitespace
        data.columns = [dc.replace(' ', '') for dc in data.columns]
        # add layer label (need to reduce on this)
        data['clayer'] = clayer
        # re-do the datetime column as it is out by 30 minutes
        data['DATE'] = pd.date_range(start=self.start_date, periods=len(data), \
                                    freq=self.time_freq)
        # set dataframe index on this column
        data.drop('Time', axis=1, inplace=True)
        # set new indices
        data_dated = data.set_index(['DATE', 'clayer'])
        # return to user
        return data_dated

    def import_spa_output(self, file_name):
        """
        Imports hourly SPA land-surface flux outputs with the index on the time
        column
        """
        # read in file
        data = pd.read_csv(file_name, sep=r',', engine='python')
        # remove header whitespace
        data.columns = [dc.replace(' ', '') for dc in data.columns]
        # re-do the datetime column as it is out by 30 minutes
        data['DATE'] = pd.date_range(start=self.start_date, periods=len(data), \
                                    freq=self.time_freq)
        # set dataframe index on this column
        data_dated = data.set_index(['DATE'])
        # return to user
        return data_dated

    def import_soil_profile(self, file_name):
        # import the soil profile input file
        soil_prof = pd.read_csv(file_name)
        # take the first row which has the soil layer thicknesses
        soil_layer = soil_prof.ix[0, 1:-1]
        # calculate the cumulative thickness, equivalent to soil depth at each layer
        cumul_depth = [soil_layer[0] + sum(soil_layer[:i]) for i in range(len(soil_layer))]
        # return to user
        return cumul_depth
    def load_gasex_raw(self, filepaths):
        """
        Special wrapper function to perform a mass import of the SPA canopy
        layer outputs from a simulation folders. The canopy layer files are
        individually imported and then aggregated into tree and grass
        components using the reduce_canopy and flatten functions. These are
        then added with the land-surface outputs to simulation ncdf files.
        """
        # Import the canopy layer output files from each model experiment
        canopy_data = [self.import_spa_canopy(fname, i) \
                        for (i, fname) in enumerate(filepaths)]

        # Reduce and flatten the canopy layers into tree and grass outputs
        treegrass_data = self.reduce_canopy(canopy_data)
        treegrass_flat = self.flatten_dictlist(treegrass_data)

        # Return to user
        return treegrass_flat

    def load_hourly_raw(self, filepaths):
        """
        Wrapper function to perform a mass import of SPA outputs from a
        simulation files using Pandas. Returns a dictionary that can be used
        later to process outputs into ncdf file.
        """
        # import the land-surface output files from each model experiment
        land_surf = {fname.split(".")[0].split("/")[-1]: \
                    self.import_spa_output(fname) \
                    for fname in filepaths}

        return land_surf

    def get_spa_filepaths(self, dir_path):
        """
        Given the top directory, find all subfolders that contain
        outputs from the SPA 1 model.

        regex_remove: controls the files that are ignored in the scan
        regex_take: controls the subfolders that are search for SPA outputs

        Returns file_list, which contains the full paths for each output
        for each SPA simulation folder.
        """
        # Files to be ignored in the search
        files_remove = ['canopy','DS_Store','ci','drivers','energy','iceprop', \
                        'parcheck','power','waterfluxes','test','solar','daily']
        # Turn list into a regex argument
        regex_remove = r'^((?!{0}).)*$'.format("|".join(files_remove))
        # Subfolders to be searched
        regex_take = r'(\binputs\b)|(\boutputs\b)$'

        # Number of subfolders
        n_subfold = len(regex_take.split("|"))

        # Walk down through the directory path looking for SPA files
        raw_list = [[os.path.join(dp, f) for f in fn \
                    if re.search(regex_remove, f)] \
                    for (dp, dn, fn) in os.walk(dir_path, followlinks=False) \
                    if re.search(regex_take, dp)]

        # Exit program if no files found
        if len(raw_list) == 0:
            # Echo user
            sys.stderr.write('No output files found. Check your FilePath. \n')
            sys.exit(-1)
        else:
            # Attach the two subfolders in each simulation together
            file_list = [natsort.natsorted(raw_list[i] + raw_list[i+1]) \
                        for i in np.arange(0, len(raw_list), n_subfold)]
            # Pass back to user
            return file_list

    def process_outputs(self, fpath_list, nc_fout):
        """
        Given a list of files from a simulation folder load them using pandas
        and process them into netcdf4 files
        """
        # Static soil profile path
        profpaths = [fp for fp in fpath_list if re.search(r'_soils', fp)][0]
        # Paths for soil profile water fluxes
        soilpaths = [fp for fp in fpath_list if re.search(r'soil(?!s)|upfrac', fp)]
        # Canopy layer outputs
        canopaths = [fp for fp in fpath_list if re.match(r'^.*[0-9]+.csv$', fp)][2:]
        # Paths for land-surface fluxes
        landpaths = [fp for fp in fpath_list \
                     if re.search(r'^((?!l\d+|soil|upfrac|temp|phen).)*$', fp)]

        # Canopy layer outputs
        canopy10 = self.load_gasex_raw(canopaths)
        # Land-surface and meteorology fluxes
        landsurf = self.load_hourly_raw(landpaths)
        # Below-ground water fluxes
        #watrprof = self.load_hourly_raw(soilpaths)
        # soil layer thicknesses
        #soil_thick = self.import_soil_profile(profpaths)

        print("> writing netCDF file: {0}".format(nc_fout))

        # Open a NCDF4 file for SPA simulation outputs
        nc_file = nc.Dataset(nc_fout, 'w', format='NETCDF4')

        # Assign attributes
        ncdf_attr = spa_nc.spa_netCDF4()
        ncdf_attr.assign_variables(nc_file)
        ncdf_attr.assign_units(nc_file)
        ncdf_attr.assign_longNames(nc_file)

        # Assign values to variables
        tseries = pd.timedelta_range(0, periods=len(landsurf['hourly']), freq="1800s") \
                    .astype('timedelta64[s]')

        # Get time from netcdf driver file
        nc_file.variables['time'][:] = np.array(tseries)
        # [Land-surface fluxes]
        nc_file.variables['GPP'][:] = np.array(landsurf['hourly']['gpp'])
        nc_file.variables['Qle'][:] = np.array(landsurf['hourly']['lemod'])
        nc_file.variables['TVeg'][:] = np.array(landsurf['hourly']['transle'])
        nc_file.variables['Esoil'][:] = np.array(landsurf['hourly']['soille'])
        nc_file.variables['Ecanop'][:] = np.array(landsurf['hourly']['wetle'])
        nc_file.variables['AutoResp'][:] = np.array(landsurf['hourly']['resp'])
        # [Vegetation fluxes]
        nc_file.variables['Atree'][:] = np.array(canopy10['trees_Ag'])
        nc_file.variables['Agrass'][:] = np.array(canopy10['grass_Ag'])
        nc_file.variables['Rtree'][:] = np.array(canopy10['trees_Rd'])
        nc_file.variables['Rgrass'][:] = np.array(canopy10['grass_Rd'])
        nc_file.variables['Etree'][:] = np.array(canopy10['trees_Et'])
        nc_file.variables['Egrass'][:] = np.array(canopy10['grass_Et'])
        nc_file.variables['Gtree'][:] = np.array(canopy10['trees_gs'])
        nc_file.variables['Ggrass'][:] = np.array(canopy10['grass_gs'])
        nc_file.variables['LAItree'][:] = np.array(canopy10['trees_LAI'])
        nc_file.variables['LAIgrass'][:] = np.array(canopy10['grass_LAI'])

        # Close file
        nc_file.close()

