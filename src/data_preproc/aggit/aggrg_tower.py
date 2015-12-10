#!/usr/bin/env python2

import os
import cPickle as pickle
from scipy import integrate

def main():
    """
    Does a quick re-sampling of flux tower observation form hourly to daily
    timesteps
    """
    tower = pickle.load(open(OBSFILE, 'rb'))

    # lambda function for integrated sum
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    tower_day = tower.resample('D', dayint)

    pickle.dump(tower_day, open(SAVFILE, 'wb'))

    return 1

if __name__ == "__main__":

    FILEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")
    OBSFILE = FILEPATH + "hourly/tower_fluxes_valid.pkl"
    SAVFILE = FILEPATH + "daily/tower_fluxes_daily.pkl"

    main()
