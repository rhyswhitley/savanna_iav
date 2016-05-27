#!/usr/bin/env python2

import pandas as pd
import pickle
import os

def main():

    tower_data = pd.read_csv(FILEPATH, index_col=[0], parse_dates=True)

    tower_fluxes = tower_data.ix[:, ["Fe", "Fe_Con", "GPP_Con", "Fre_Con", "Fc", "Fc_ustar"]]

    with open(SAVEPATH, 'wb') as handle:
        pickle.dump(tower_fluxes, handle)

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Dropbox/Dingo12/Advanced_processed_data_HowardSprings_v12.csv")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/tower_fluxes_valid.pkl")

    main()
