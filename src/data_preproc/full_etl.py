#!/usr/bin/env python2

import re
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

    print("Starting ETL processes on SPA v1 outputs")

    # Create a SPA ETL object
    spa_etl = spa.SPAoutput_ETL()

    print("The script will now convert raw CSV outputs into netCDF4 files\n")

    # Get the directory and file path information
    file_path_list = spa_etl.get_spa_filepaths(DIRPATH)

    # Load outputs files and process into dictionaries of dataframes
    save_paths = ["{0}/spa_hws_exp{1}.nc".format(SAVEPATH, i + 1) \
                    for i in range(len(file_path_list))]

    # Upload files and transfer to a netCDF format
    [spa_etl.process_outputs(flist, sp) \
        for (flist, sp) in zip(file_path_list, save_paths)]

    print("ETL Finished")


if __name__ == "__main__":

    # Filepaths set here
    DIRPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/")
    SAVEPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf")

    # Run main
    main()


