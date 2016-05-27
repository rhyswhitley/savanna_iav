#!/usr/bin/env python

import os
import cPickle as pickle
from scipy import integrate
from collections import OrderedDict
from natsort import natsorted

def main():

    # load pickle object
    flux_dict = pickle.load(open(PKLPATH + "hourly/fluxes_dict.pkl", 'rb'))

    # lambda function for integrated sum
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    dict_samp = {lab: dayint if i in range(4)+[6] else 'mean' \
                 for (i, lab) in enumerate(flux_dict['Exp_1'].columns) }

    # resample this to daily data
    daily_fluxes_df = {dlab: df.resample('D', how=dict_samp) \
                       for (dlab, df) in flux_dict.iteritems()}

    # dump this as another pickle for plotting later on
    pickle.dump(OrderedDict(natsorted(daily_fluxes_df.iteritems())), \
                open(PKLPATH + "daily/daily_fluxes.pkl", 'wb'))

    return 1

if __name__ == '__main__':

    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
