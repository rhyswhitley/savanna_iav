#!/usr/bin/env python

import os
import cPickle as pickle
from scipy import integrate

def reduce_to_day(df):

    # lambda function for integrated sum
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    # columns 4-5 are conductance which isn't summed
    up_samps = {clab: dayint \
                if i < len(df.columns) - 1 else 'mean' \
                for (i, clab) in enumerate(df.columns)}

    # resample based on the up-scaling dict above
    daily_df = df.bfill().resample('D', how=up_samps)

    return daily_df

def main():

    metdata = pickle.load(open(PKLPATH + "hourly/tower_fluxes_valid.pkl", 'rb'))

    # daily aggregation here
    daily_met = reduce_to_day(metdata)

    pickle.dump(daily_met, open(PKLPATH + "daily/daily_tower_fluxes.pkl", 'wb'))

    return 1

if __name__ == '__main__':

    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
