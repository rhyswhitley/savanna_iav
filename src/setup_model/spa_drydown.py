#!/usr/bin/env python2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # import the file
    raw_file = pd.read_csv(FILEPATH)
    # add an index ** needed to manipulate rainfall
    raw_file['DT'] = pd.date_range("2001-01-01", periods=len(raw_file), freq='30min')
    # set that index
    raw_file.set_index(['DT'], inplace=True)
    # pick from which year you wish there to be no rain
    raw_file.ix[raw_file.index.year > 2001, "Precip_Con"] = 0.0

    # there is a major issue of 17 days of missing energy data - throws off the simulation
    # need to fix in original experiemnts
    raw_file.ix["2001-10-05":"2001-10-21", "Fsd_Con"] = np.tile(raw_file["2001-10-04"].Fsd_Con.values, 17)
    raw_file.ix["2001-10-05":"2001-10-21", "PAR"] = np.tile(raw_file["2001-10-04"].PAR.values, 17)

    # write this to the dry down experiment folder
    raw_file.to_csv(OUTPATH, index=False)

    return None

if __name__ == "__main__":

    EXPN = 1
    FILE = ("~/Savanna/Models/SPA1/outputs/site_co2/" \
            "HS_Exp{0}/inputs/hs_met_exp_{0}.csv".format(EXPN))
    FILEPATH = os.path.expanduser(FILE)

    OUTNAME = "hs_met_nowater.csv"
    OUTDIR = os.path.expanduser("~/Savanna/Models/SPA1/outputs/drydown/HowardSprings/inputs/")
    OUTPATH = OUTDIR + OUTNAME

    main()
