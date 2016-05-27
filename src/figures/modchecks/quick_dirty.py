#!/usr/bin/env python2

import numpy as np
import natsort
import pandas as pd
import os, re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import expanduser
from scipy import integrate
from pylab import get_cmap

def import_spa_canopy(file_name, spaf):
    """
    Imports SPA canopy layer outputs into a pandas df
    > indexes on time and canopy layer
    """
    # read in file
    data = pd.read_csv(file_name, sep=r',\s+', engine='python')
    data.columns = [dc.replace(' ', '') for dc in data.columns]
    # re-do the datetime column as it is out by 30 minutes
    data['DATE'] = pd.date_range(start="2001-01-01", periods=len(data), freq=spaf)
    # set dataframe index on this column
    #data.drop('Time', axis=1, inplace=True)
    data_dated = data.set_index('DATE')
    # return to user
    return data_dated

def main():

    print "Importing data from: {0}".format(DIRPATH)
    # get the output file of interest from each of the experiment folders
    path_lists = natsort.natsorted([os.path.join(dp, f) \
                    for (dp, dn, fn) in os.walk(DIRPATH) \
                    for f in fn \
                    if re.search(r'(^l.*csv)', f)])

    print "Loading data ..."
    canopy_data = [import_spa_canopy(exp_paths, '30min') for exp_paths in path_lists]

    col1_map = get_cmap("rainbow")(np.linspace(0, 1, len(canopy_data)))

    plt.figure(figsize=(10, 8))
    mygrid = gridspec.GridSpec(5, 2, hspace=1)

    for (i, cdata) in enumerate(canopy_data):
        ax = plt.subplot(mygrid[i])
        gpp = cdata['Ag'].resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6*12)
        ax.plot(gpp, '-', lw=1.5, c=col1_map[i], label='layer {0}'.format(i+1))
        ax.set_title('layer {0}'.format(i+1))
    #plt.legend(loc='upper center', bbox_to_anchor=(1.0, 1),prop={'size':10})
    plt.show()

    print "Done"
    return None

if __name__ == "__main__":

    DIRPATH = expanduser("~/Savanna/Models/SPA1/outputs/site_co2/HS_Exp1/outputs/")

    main()
