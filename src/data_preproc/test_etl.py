#!/usr/bin/env python3

import os
import time

from etl import SPAoutput_ETL

__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = "2015-10-15"
__modified__ = time.strftime("%c")
__version__ = '1.0'
__status__ = 'prototype'

def main():

    print("Test the canopy reduction process")

    # Create a SPA ETL object
    spa = SPAoutput_ETL()
    print(spa.start_date)
    print(spa.time_freq)

    return 1


if __name__ == "__main__":

    # Filepaths set here
    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf")

    # Run main
    main()


