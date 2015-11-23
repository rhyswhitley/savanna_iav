#!/usr/bin/env python2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os, re
from os.path import expanduser


def main():

    # get the output file of interest from each of the experiment folders
    folders = [[os.path.join(dp, f) for f in fn if re.search(r'(^l.*csv)', f)] \
               for (dp, dn, fn) in os.walk(DIRPATH) if re.search(r'(.*outputs_exp)', dp)]

    for fold in folders:
        print fold


    return None

if __name__ == "__main__":

    DIRPATH = expanduser("~/Savanna/Models/SPA1/outputs/site_co2/HowardSprings/")

    main()
