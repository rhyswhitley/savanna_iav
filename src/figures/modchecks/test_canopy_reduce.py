#!/usr/bin/env python3

import pandas as pd
import os, re
import time

import seaborn as sns
import matplotlib.pyplot as plt

from etl import SPAoutput_ETL

__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = "2015-10-15"
__modified__ = time.strftime("%c")
__version__ = '1.0'
__status__ = 'prototype'

def main():

    print("Test the canopy reduction process")

    # Create a SPA ETL object
    spa = SPAoutput_ETL()

    # Get the directory and file path information
    fpath_list = spa.get_spa_filepaths(DIRPATH)[1]

    for fp in fpath_list:
        if re.match(r'^.*_phen_+.csv', os.path.basename(fp)):
            print(fp)
    print("")

    # check LAI from canopy outputs
    phen = pd.read_csv(fpath_list[2], sep=r',')
    phen['DT'] = pd.date_range(start="2001-01-01", freq='D', periods=len(phen))
    phen['c3frac'] = phen.ix[:, 3:8].sum(axis=1)
    phen['c4frac'] = phen.ix[:, 8:13].sum(axis=1)
    phen['c3lai'] = phen['lai']*phen['c3frac']
    phen['c4lai'] = phen['lai']*phen['c4frac']
    phen.set_index(['DT'], inplace=True)

    canopaths = [fp for fp in fpath_list if \
                 #re.search(r'_soils', os.path.basename(fp))]
                 re.search(r'l.*[0-9]+.csv$', os.path.basename(fp))]
    for cp in canopaths:
        print(cp)

    return 1

    leaf = spa.load_gasex_raw(canopaths[2:])
    leaf_d = leaf.resample('D').apply('mean')

    fig = plt.figure(figsize=(11, 7))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    ax1.plot(phen["lai"], 'g-', label='input')
    ax1.plot(leaf_d["trees_LAI"] + leaf_d["grass_LAI"], \
                                    c='darkgreen', lw=2, alpha=0.5, label='output')

    ax2.plot(phen["c4lai"], 'r-', label='input')
    ax2.plot(leaf_d["grass_LAI"], '-', c='darkred', lw=2, alpha=0.5, label='output')

    ax3.plot(phen["c3lai"], 'b-', label='input')
    ax3.plot(leaf_d["trees_LAI"], '-', c='darkblue', lw=2, alpha=0.5, label='output')

    ax1.legend(loc="upper center", ncol=2)
    ax2.legend(loc="upper center", ncol=2)
    ax3.legend(loc="upper center", ncol=2)

    ax1.set_ylim(0, 3)
    ax2.set_ylim(0, 1.5)
    ax3.set_ylim(0, 1.5)

    plt.show()

    return 1


if __name__ == "__main__":

    # Filepaths set here
    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf")

    # Run main
    main()


