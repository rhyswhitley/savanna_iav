#!/usr/bin/env python

import os
import cPickle as pickle
from scipy import integrate
from collections import OrderedDict

def main():

    # load pickle object
    flux_dict = pickle.load(open(PKLPATH + "hourly/fluxes_dict.pkl", 'rb'))

    # lambda function for integrated sum
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    # resample this to daily data
    daily_fluxes_df = {dlab: df.resample('D', how=dayint) \
                       for (dlab, df) in flux_dict.iteritems()}

    # dump this as another pickle for plotting later on
    pickle.dump(OrderedDict(sorted(daily_fluxes_df.iteritems())), \
                open(PKLPATH + "daily/daily_fluxes.pkl", 'wb'))

    return 1

if __name__ == '__main__':

    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
