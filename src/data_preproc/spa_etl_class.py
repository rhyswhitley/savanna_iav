import natsort
import pandas as pd
import os, re, cPickle
import time
from joblib import Parallel, delayed

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
        self.dirpath = None
        self.savepath = None
        self.soil_depth = None

    def reduce_canopy(self, data_list, gtl=6):
        """
        Generalise the 10 canopy layer files into a dictionary of tree and
        grass dataframes
        """
        # split layer files into trees and grasses and
        # concatenate these together
        trees_raw = pd.concat(data_list[:gtl], axis=0)
        grass_raw = pd.concat(data_list[gtl:], axis=0)

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

    def clean_hourly_header(self, old_df):
        """Replaces default hourly headers with a cleaner version"""
        # new headers
        new_headers = ["GPP", "Resp", "LE_total", "LE_veg", "LE_soil", "LE_wet", "Lambda"]
        # remove first and last files
        new_df = old_df.ix[:, 1:-1]
        # add new headers
        new_df.columns = new_headers
        # return new data.frame
        return new_df

    def clean_swc_headers(self, old_df, soil_depths):
        """Replaces default soil water content headers with a cleaner version"""
        new_headers = ["swc_{0}cm".format(depth) for depth in soil_depths]
        new_df = old_df.ix[:, 1:]
        new_df.columns = new_headers + ["SWP_all"]
        return new_df

    def clean_upfr_headers(self, old_df, soil_depths, label):
        """Replaces default water uptake headers with a cleaner version"""
        new_headers = ["{0}_{1}cm".format(label, depth) for depth in soil_depths]
        new_df = old_df.ix[:, 1:]
        new_df.columns = new_headers
        return new_df

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
        self.soil_depth = cumul_depth
    def load_gasex_data(self):

        # Get the output file of interest from each of the experiment folders
        path_lists = [natsort.natsorted([os.path.join(dp, f) for f in fn \
                        if re.search(r'(^l.*csv)', f)]) \
                        for (dp, dn, fn) in os.walk(self.dirpath) \
                        if re.search(r'(outputs$)', dp)]

        # create parallel versions of functions
        #xp_imp_spa = delayed(self.import_spa_canopy)
        #xp_red_can = delayed(self.reduce_canopy)

        print "Loading data ..."
        # Import SPA output files
        canopy_data = [[self.import_spa_canopy(e_p, i) for e_p in exp_paths] \
                        for (i, exp_paths) in enumerate(path_lists)]

        print "Reducing data ..."
        # Reduce the 10 layer output into generalised tree and grass outputs
        treegrass_data = [self.reduce_canopy(cdata) for cdata in canopy_data]

        # For each experiment flatten each dictionary of tree/grass outputs
        # into one dataframe
        data_flat = {"exp_{0}".format(i + 1): self.flatten_dictlist(ex_df) \
                        for (i, ex_df) in enumerate(treegrass_data)}

        # Pickle dataset
        with open(self.savepath + "spa_treegrass_output.pkl", 'wb') as handle:
            cPickle.dump(data_flat, handle)

        return data_flat

    def load_hourly_data(self, otype='hourly'):

        # get the output file of interest from each of the experiment folders
        folders = [os.path.join(dp, f) \
                for (dp, dn, fn) in os.walk(self.dirpath) \
                for f in fn \
                if re.search(otype, f)]

        print "Loading data ..."
        # import hourly output files from each model experiment
        raw_data = [self.import_spa_output(fname) \
                    for fname in folders]

        # save the list of dataframes as a dictionary
        data_frame = {"exp_{0}".format(i+1): cdata \
                        for (i, cdata) in enumerate(raw_data)}

        # save as a pickle object
        pkl_savefile = "{0}spa_{1}_output.pkl" \
            .format(self.savepath, otype)
        with open(pkl_savefile, 'wb') as handle:
            cPickle.dump(data_frame, handle)

        return data_frame

#
#        # clean-up dataframes based on output type
#        if otype is "hourly":
#            clean_data = [self.clean_hourly_header(rd) \
#                          for rd in raw_data]
#        elif otype is "soilwater":
#            clean_data = [self.clean_swc_headers(rd, self.soil_depth) \
#                          for rd in raw_data]
#        else:
#            clean_data = [self.clean_upfr_headers(rd, self.soil_depth, "upfrac") \
#                          for rd in raw_data]
#

#        # Write to csv file as well
#        for (tg_lab, tg_df) in tg_data.iteritems():
#            print "Writing SPA gasex file for {0}".format(tg_lab)
#            tg_df.to_csv("{0}{1}-gasex_HWS.csv".format(self.savepath, tg_lab))

#        # write to csv
#        for (i, cdata) in enumerate(clean_data):
#            print "Writing SPA {0} file for Exp_{1}".format(otype, i+1)
#            csv_savefile = "{0}/exp_{1}-{2}_HWS.csv" \
#                .format(self.savepath, i+1, otype)
#            cdata.to_csv(csv_savefile)
