#!/usr/bin/env python

import os
import cPickle as pickle

    # lambda function for aggregating
def reduce_to_month(df):

    agg_df = df.groupby([df.index.year, df.index.month, df.index.hour]) \
                        .mean()
    agg_df.index.names = ["year", "month", "hour"]

    return agg_df

def main():

    leaf_dict = pickle.load(open(PKLPATH + "hourly/leaf_dict.pkl", 'rb'))

    # aggregatin the leaf level data into mean monthly diurnal cycles
    # we don't need every day!


    # aggregate here
    mean_month_df = {dlab: reduce_to_month(df) for (dlab, df) in leaf_dict.iteritems()}

    # dump this as another pickle for plotting later on
    pickle.dump(mean_month_df, open(PKLPATH + "agg/mean_monthly_leaf.pkl", 'wb'))

    return 1

if __name__ == '__main__':

    PKLPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/")

    main()
