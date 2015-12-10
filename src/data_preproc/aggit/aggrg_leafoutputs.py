#!/usr/bin/env python

import os
import cPickle as pickle
from scipy import integrate
from collections import OrderedDict
from natsort import natsorted

    # lambda function for aggregating
def reduce_to_month(df):

    agg_df = df.groupby([df.index.year, df.index.month, df.index.hour]) \
                        .mean()
    agg_df.index.names = ["year", "month", "hour"]

    return agg_df

def reduce_to_day(df):

    # lambda function for integrated sum
    dayint = lambda x: integrate.trapz(x, dx=1800)*1e-6

    # columns 4-5 are conductance which isn't summed
    up_samps = {clab: 'mean' if i in range(4, 8) else dayint \
                for (i, clab) in enumerate(df.columns)}

    # resample based on the up-scaling dict above
    daily_df = df.bfill().resample('D', how=up_samps)

    return daily_df

def main():

    leaf_dict = pickle.load(open(PKLPATH + "hourly/leaf_dict.pkl", 'rb'))

    # aggregatin the leaf level data into mean monthly diurnal cycles
    # we don't need every day!

    # daily aggregation here
    daily_df = {dlab: reduce_to_day(df) for (dlab, df) in leaf_dict.iteritems()}

    # monthly mean hour aggregate here
    mean_month_df = {dlab: reduce_to_month(df) for (dlab, df) in leaf_dict.iteritems()}

    # dump mean-month-hour and daily dataframes for plotting later
    pickle.dump(OrderedDict(natsorted(mean_month_df.iteritems())), \
                open(PKLPATH + "agg/mean_monthly_leaf.pkl", 'wb'))

    pickle.dump(OrderedDict(natsorted(daily_df.iteritems())), \
                open(PKLPATH + "daily/daily_leaf.pkl", 'wb'))

    return 1

if __name__ == '__main__':

    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
