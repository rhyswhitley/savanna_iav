#!/usr/bin/env python2

import os
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.gridspec as gridspec

def plot_inputs(dataset, phen, EX=1):

    n_plots = 6
    gs = gridspec.GridSpec(n_plots, 1)
    ax1 = [plt.subplot(gs[i]) for i in range(n_plots)]

    # turn off x labels
    for (i, subax) in enumerate(ax1):
        if i < len(ax1) - 1:
            subax.axes.get_xaxis().set_visible(False)

    # plots
    time_x = pd.date_range(start="2001-01-01", end="2014-12-31", freq='D') #.to_pydatetime()
    ax1[0].plot_date(time_x, dataset["SWdown"].resample('D', how=lambda x: integrate.trapz(x, dx=1800)*1e-6), '-', c='red')
    ax1[1].plot_date(time_x, dataset["VPD"].resample('D', how='mean'), '-', c='purple')
    ax1[2].plot_date(time_x, dataset["Tair"].resample('D', how='mean'), '-', c='orange')
    ax1[3].plot_date(time_x, dataset["Wind"].resample('D', how='mean'), '-', c='darkgreen')
    ax1[4].plot_date(time_x, dataset["Rainfall"].resample('D', how='sum'), '-', c='blue')
    ax2 = ax1[4].twinx()
    ax2.plot_date(time_x, dataset["CO2"].resample('D', how='mean'), '-', c='black', lw=2)
    ax1[5].plot_date(time_x, dataset["LAI"], '-', c='green')

    # labels
    plt_label = "Howard Springs Experiment {0} Inputs".format(EX)
    ax1[0].set_title(plt_label)
    ax1[0].set_ylabel("$R_{s}$ (MJ m$^{-2}$)")
    ax1[1].set_ylabel("$D_{v}$ (kPa)")
    ax1[2].set_ylabel("$T_{a}$ ($\degree$C)")
    ax1[3].set_ylabel("$U_{v}$ (m s$^{-1}$)")
    ax1[4].set_ylabel("PPT (mm)")
    ax1[5].set_ylabel("LAI")
    ax2.set_ylabel("CO$_{2}$ (ppm)")
    for i in range(len(ax1)):
        ax1[i].yaxis.set_label_coords(-0.07, 0.5)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.1)
    #plt.savefig(figure_path + plt_label.replace(' ', '_') + ".pdf", rasterized=True)
    plt.show()

    return None

def get_value(nc_obj, label):
    return nc_obj.variables[label][:].flatten()

def main():

    nc_conn = nc.Dataset(DIRPATH, 'r', format="NETCDF4")
    time_len = len(nc_conn.dimensions['time'])

    tower_df = pd.DataFrame({ \
                            "SWdown": get_value(nc_conn, "SWdown"), \
                            "VPD": get_value(nc_conn, "VPD"), \
                            "Tair": get_value(nc_conn, "Tair"), \
                            "Cair": get_value(nc_conn, "Cair"), \
                            "Wind": get_value(nc_conn, "Wind"), \
                            "Rainfall": get_value(nc_conn, "Rainfall"), \
                            "LAI": get_value(nc_conn, "LAI"), \
                             },
                            index=pd.date_range("2001-01-01 00:30:00", \
                            periods=time_len, freq="30min"))
    print tower_df.head(10)


    return None

if __name__ == "__main__":

    DIRPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/ncdf/spa_hws_inputs.nc")

    main()
