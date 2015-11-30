#!/usr/bin/env python

import os
from collections import OrderedDict
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap
from matplotlib import style
from scipy import stats
from scipy import integrate

def plot_monthly_response(norm, pert):

    plot_grid = gridspec.GridSpec(4, 1, hspace=0.1)

    ax1 = plt.subplot(plot_grid[0])
    ax2 = plt.subplot(plot_grid[1])
    ax3 = plt.subplot(plot_grid[2])
    ax4 = plt.subplot(plot_grid[3])

    # Stomatal conductance
    ax1.plot(norm["Gtree"].values)
    ax1.plot(pert["Gtree"].values)

    # Leaf transpiration
    ax2.plot(norm["Etree"].values)
    ax2.plot(pert["Etree"].values)

    # Leaf assimilation
    ax3.plot(norm["Atree"].values)
    ax3.plot(pert["Atree"].values)

    ax4.plot(norm["LAItree"].values)
    ax4.plot(pert["LAItree"].values)
    ax4.plot(norm["LAIgrass"].values)
    ax4.plot(pert["LAIgrass"].values)

    plt.show()

    return 1

def main():

    data_dict = pickle.load(open(PKLPATH, 'rb'))

    year_agg = lambda x: x.groupby(level=['month', 'hour']).mean()

    data_mean_year = [year_agg(df) \
                      for df in OrderedDict(data_dict).values()]

    # **FOR LOOP WILL GO HERE
    plot_monthly_response(data_mean_year[3], data_mean_year[6])

    return 1

if __name__ == "__main__":

    FILEPATH = "~/Savanna/Data/HowardSprings_IAV/pickled/agg/mean_monthly_leaf.pkl"
    PKLPATH = os.path.expanduser(FILEPATH)

    main()
