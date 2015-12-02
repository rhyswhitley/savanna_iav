#!/usr/bin/env python2

import pandas as pd
import os
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import style

def plot_wue_time(df):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = plt.rcParams['axes.color_cycle']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(df.index, df["WUE"], color=ncols[0], width=300, \
            edgecolor=ncols[0], alpha=0.7)
    ax1.xaxis_date()
    ax2.plot_date(df.index, df["dWUE_dt"], '-', c=ncols[1], lw=3)

    ax1.set_ylabel(r'WUE (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')
    ax2.set_ylabel(r'$\partial$WUE/$\partial$t (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')

    ax1.set_ylim([0, 3.2])
    ax2.set_ylim([-1, 1])

    ax2.grid(False)

    plt.show()

    return 1

def plot_wue_co2(df):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = plt.rcParams['axes.color_cycle']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df['Cair'], df["WUE"], '-', c=ncols[0], alpha=0.7)
    ax2.plot(df['Cair'], df["dWUE_dt"], '-', c=ncols[1], lw=3)

    ax1.set_xlabel(r'C$_{air}$ (ppm)')
    ax1.set_ylabel(r'WUE (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')
    ax2.set_ylabel(r'$\partial$WUE/$\partial$t (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')

    ax1.set_ylim([0, 3.2])
    ax2.set_ylim([-1, 1])

    ax2.grid(False)

    plt.show()

    return 1

def plot_wue_ppt(df):

    plt.rcParams['lines.linewidth'] = 1.25
    plt.rcParams.update({'mathtext.default': 'regular'})
    style.use('ggplot')
    ncols = plt.rcParams['axes.color_cycle']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df['Rainfall'], df["WUE"], 'o', c=ncols[0], alpha=0.7)
    ax2.plot(df['Rainfall'], df["dWUE_dt"], 'o', c=ncols[1], lw=3)

    ax1.set_xlabel(r'Rainfall (mm)')
    ax1.set_ylabel(r'WUE (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')
    ax2.set_ylabel(r'$\partial$WUE/$\partial$t (mol H$_{2}$O mol$^{-1}$ CO$_{2}$)')

    ax1.set_ylim([0, 3.2])
    ax2.set_ylim([-1, 1])

    ax2.grid(False)

    plt.show()

    return 1

def main():
    """
    Does a quick re-sampling of flux tower observation form hourly to daily
    timesteps
    """
    pload = lambda x: pickle.load(open(x, 'rb'))

    tower = pload(OBSFILE)
    meteo = pload(INFILE)
    model_dict = pload(OBSFILE)

    observed = pd.concat([tower[["Fe_Con", "GPP_Con"]], \
                          meteo[["Cair", "Rainfall"]]], axis=1)

    samp_dict = {lab: 'sum' if i is not 2 else 'mean' \
                 for (i, lab) in enumerate(observed.columns)}
    tower_year = observed.resample('A', samp_dict)

    # convert LE to mols
    tower_year["Fe_mol"] = tower_year["Fe_Con"]/18/2.45
    tower_year["WUE"] = -tower_year["GPP_Con"].divide(tower_year["Fe_mol"])
    tower_year["dWUE_dt"] = tower_year["WUE"].diff()

    print tower_year

    plot_wue_ppt(tower_year)

    return 1


if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    INFILE = FILEPATH + "daily_inputs.pkl"
    OBSFILE = FILEPATH + "tower_fluxes_daily.pkl"
    MODFILE = FILEPATH + "daily_fluxes.pkl"

    main()
