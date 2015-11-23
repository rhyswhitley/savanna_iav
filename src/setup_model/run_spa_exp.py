#!/usr/bin/env python2

import os, re
import subprocess as sub

def main():

    binary_locs = [os.path.join(dp, f) for (dp, fp, fn) \
                in os.walk(DIRPATH) for f in fn \
                if re.match('SPA', f)]

    p = [sub.Popen(spa_bin) for spa_bin in binary_locs]
    #p = [sub.Popen(spa_bin, stdout=sub.PIPE, stderr=sub.PIPE) for spa_bin in binary_locs]

    return 1

if __name__ == "__main__":

    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")

    main()

