#!/usr/bin/env python2

import spa_etl_class as spa
import os
import time

__author__ = 'Rhys Whitley'
__email__ = 'rhys.whitley@gmail.com'
__created__ = "2015-10-15"
__modified__ = time.strftime("%c")
__version__ = '1.0'
__status__ = 'prototype'

def main():

    print "Starting ETL processes on SPA v1 outputs"

    # Create a SPA ETL object
    spa_etl = spa.SPAoutput_ETL()

    # Set ETL attributes
    # ----------------------------------------------------------------------
    # Directory attributes
    spa_etl.dirpath = DIRPATH
    spa_etl.savepath = SAVEPATH
    # Set soil depth attribute
    spa_etl.import_soil_profile(SOILPATH)
    # Load outputs files and process into dictionaries of dataframes
    gasex = spa_etl.load_gasex_data()
    bulkflux = spa_etl.load_hourly_data("hourly")
    swc = spa_etl.load_hourly_data("soilwater")
    upfrac = spa_etl.load_hourly_data("upfrac")

    print "ETL Finished"


if __name__ == "__main__":

    # Filepaths set here
    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/")
    SOILPATH = DIRPATH + "HS_Exp1/inputs/HowardSprings_soils.csv"

    # Run main
    main()


