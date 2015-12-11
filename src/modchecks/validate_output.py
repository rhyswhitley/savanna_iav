#!/usr/bin/env python2

import os
import numpy as np
import pandas as pd
import cPickle as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy import integrate
import sklearn.metrics as metrics


def fit_trend(x, y):
    z = np.polyfit(x, y, 1)
    return np.poly1d(z)

def draw_trend(pObj, x, y, *args, **kwargs):
    if len(x) > len(y):
        return add_trend(pObj, x[:len(y)], y, *args, **kwargs)
    else:
        return add_trend(pObj, x, y[:len(x)], *args, **kwargs)

def add_trend(pObj, xseries, yseries, *args, **kwargs):
        # get linear trend line
        trend = fit_trend(xseries, yseries)
        # plot trend line
        pObj.plot(xseries, trend(xseries), *args, **kwargs)

def rmse(obs, pred):
    return(np.sqrt(np.mean((obs - pred)**2)))

def nme(obs, pred):
    '''PLUMBER normalised mean error'''
    return(np.sum(np.abs(obs - pred))/np.sum(np.abs(obs - np.mean(obs))))

def mbe(obs, pred):
    '''PLUMBER mean bias error'''
    return(np.sum(pred - obs)/len(obs))

def sd_diff(obs, pred):
    '''PLUMBER standard deviation difference'''
    return(abs(1 - np.std(pred)/np.std(obs)))

def get_metrics(obs, mod):
    # correlation coefficient
    r2_ = np.corrcoef(mod, obs)[0, 1]
    # coefficient of determination
    me_ = metrics.r2_score(obs, mod)
    # normalised mean error
    nme_ = nme(obs, mod)
    # mean bias error
    mbe_ = mbe(obs, mod)
    # root mean square error
    sd_ = sd_diff(obs, mod)
    # intercept and slope (in that order)
    #lnfit = sm.OLS(np.array(obs), sm.add_constant(np.array(mod)))
    # attach lists and return to user
    return {'r2': r2_, 'nme': nme_, 'sd': sd_, 'mbe': mbe_} # + list(lnfit.fit().params)

def summary_table(x, y):

    if len(x) > len(y):
        return get_metrics(abs(x[:len(y)]), abs(y))
    else:
        return get_metrics(abs(x), abs(y[:len(x)]))

def import_spa_output(file_name, start_date, dfreq="30min"):
    """
    Imports dingo meteorology file and puts an index on the time column
    """
    # read in file
    data = pd.read_csv(file_name, sep=r',\s+', engine='python', na_values=["Infinity"])
    data.columns = [dc.replace(' ', '') for dc in data.columns]
    # re-do the datetime column as it is out by 30 minutes
    data['DATE'] = pd.date_range(start=start_date, periods=len(data), freq=dfreq)
    # set dataframe index on this column
    data_dated = data.set_index(['DATE'])
    # return to user
    return data_dated.bfill()

def get_bound(xs_grp, upper=True):
    if upper is True:
        return xs_grp.mean() + xs_grp.std()
    else:
        return xs_grp.mean() - xs_grp.std()

def bound_plot(obj, x, y, *args, **kwargs):
    return obj.fill_between(x, \
        get_bound(y, True), get_bound(y, False), \
        *args, **kwargs)

def unplot(obj, x, y, *args, **kwargs):

    if len(x) > len(y):
        newx = x[:len(y)]
        obj.plot(newx, y, *args, **kwargs)
    else:
        newy = y[:len(x)]
        obj.plot(x, newy, *args, **kwargs)
    return None


def main():
    """Checks whether SPA outputs are matching observations"""

    hws_obs = pickle.load(open(DATFILE, 'rb'))
    spa_out = import_spa_output(SPAFILE, hws_obs.index[0])

    # integrate to daily time-step
    spa_day = spa_out.resample('D', how=lambda x: integrate.trapz(x, dx=1800)/1e6)

    hws_obs['GPPgC'] = hws_obs['GPP']*12
    spa_day['GPPgC'] = spa_day['gpp']*12

    # PLOTTING

    # setup
    sns.set_style("dark")
    plt.rcParams.update({'mathtext.default': 'regular'})
    fig = plt.figure(figsize=(13, 9))

    grids = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.25, width_ratios=[2, 1])
    ax1a = plt.subplot(grids[0, :])
    ax1b = ax1a.twinx()
    ax2a = plt.subplot(grids[1, 0])
    ax2b = ax2a.twinx()
    ax3a = plt.subplot(grids[1, 1])
    ax3b = ax3a.twinx()
    ax3c = ax3a.twiny()


    smooth7 = lambda x: pd.rolling_mean(x, 7, min_periods=1)
    kws = dict(lw=1.2, alpha=0.9)

    # PLOT [1]
    ax1a.plot_date(hws_obs.index, smooth7(hws_obs['Qle']), '-', c='darkblue', **kws)
    ax1a.plot_date(spa_day.index, smooth7(spa_day['lemod']), '-', c='#6495ED', **kws)
    ax1b.plot_date(hws_obs.index, smooth7(hws_obs["GPPgC"]), '-', c='darkred', **kws)
    ax1b.plot_date(spa_day.index, smooth7(spa_day['GPPgC']), '-', c='#FA8072', **kws)

    # labels
    ax1a.set_title("SPA validation:: 7-Day Running Mean, Howard Springs (2001 to 2015)", fontsize=13)
    ax1a.set_ylabel(r'LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=12, y=0.75)
    ax1b.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=12, y=0.25)

    # axis
    ax1a.yaxis.set_ticks(np.arange(0, 16, 2))
    ax1b.yaxis.set_ticks(np.arange(-16, 2, 2))
    new_dates = pd.date_range("2001", periods=15, freq='AS')
    ax1a.xaxis.set_ticks(new_dates)
    ax1a.xaxis.set_ticklabels(new_dates, rotation=0, ha="center", fontsize=12)
    ax1a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # limits
    ax1a.set_ylim([-15, 15])
    ax1b.set_ylim([-15, 15])

    ax1a.grid('on', c='white')
    ax1b.yaxis.grid(True, c='white')


    # PLOT [2]
    grp_year = lambda x: x.groupby([x.index.month, x.index.day])

    one_year = pd.date_range("2004-01-01", periods=366, freq='D')

    obs_grp = grp_year(hws_obs.ix[:, ['Qle', 'GPPgC']])
    mod_grp = grp_year(spa_day.ix[:, ['lemod', 'GPPgC']])

    ax2a.plot(one_year, obs_grp['Qle'].mean().values, '-', c='darkblue')
    ax2b.plot(one_year, obs_grp['GPPgC'].mean().values, '-', c='darkred')
    bound_plot(ax2a, one_year, obs_grp['Qle'], facecolor='gray', alpha=0.3, edgecolor='')
    bound_plot(ax2b, one_year, obs_grp['GPPgC'], facecolor='gray', alpha=0.3, edgecolor='')

    ax2a.plot(one_year, mod_grp['lemod'].mean().values, '-', c='blue', **kws)
    ax2b.plot(one_year, mod_grp['GPPgC'].mean().values, '-', c='red', **kws)
    bound_plot(ax2a, one_year, mod_grp['lemod'], facecolor='#6495ED', alpha=0.8, edgecolor='')
    bound_plot(ax2b, one_year, mod_grp['GPPgC'], facecolor='#FA8072', alpha=0.8, edgecolor='')
    # labels
    ax2a.set_title('Typical seasonality', fontsize=13)
    ax2a.set_ylabel(r'LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=12, y=0.75)
    ax2b.set_ylabel(r'GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=12, y=0.25)
    # axis
    ax2a.yaxis.set_ticks(np.arange(0, 16, 2))
    ax2b.yaxis.set_ticks(np.arange(-16, 2, 2))
    new_dates2 = pd.date_range("2004", periods=13, freq='MS')
    ax2a.xaxis.set_ticks(new_dates2)
    ax2a.xaxis.set_ticklabels(new_dates2, rotation=0, ha="center", fontsize=12)
    ax2a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # limits
    ax2a.set_ylim([-12, 15])
    ax2b.set_ylim([-12, 15])
    # grid
    ax2a.grid('on', c='white')
    ax2b.yaxis.grid(True, c='white')



    # PLOT [3]
    kws = dict(markeredgecolor='none', alpha=0.2, lw=1)
    unplot(ax3a, hws_obs['Qle'], spa_day['lemod'], 'o', c='#6495ED', **kws)
    unplot(ax3b, hws_obs['GPPgC'], spa_day['GPPgC'], 'o', c='#FA8072', **kws)
    draw_trend(ax3a, hws_obs['Qle'], spa_day['lemod'], '-', c='darkblue')
    draw_trend(ax3b, hws_obs['GPPgC'], spa_day['GPPgC'], '-', c='darkred')
    ax3c.plot([-100, 100], [-100, 100], '--', c='white', lw=1)
    # limits
    ax3a.set_xlim([-16, 16])
    ax3a.set_ylim([-16, 16])
    ax3b.set_xlim([-16, 16])
    ax3b.set_ylim([-16, 16])
    ax3c.set_xlim([-16, 16])
    # labels
    ax3b.set_ylabel(r'Tower LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=11, y=0.75)
    ax3a.set_ylabel(r'Model GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=11, y=0.25)
    ax3a.set_xlabel(r'Tower GPP (gC m$^{-2}$ d$^{-1}$)', fontsize=11, x=0.25)
    ax3c.set_xlabel(r'Model LE (MJ m$^{-2}$ d$^{-1}$)', fontsize=11, x=0.75)
    # axis
    ax3a.yaxis.set_ticks(np.arange(-16, 2, 2))
    ax3a.xaxis.set_ticks(np.arange(-16, 2, 2))
    ax3b.yaxis.set_ticks(np.arange(0, 17, 2))
    ax3c.xaxis.set_ticks(np.arange(0, 17, 2))
    # grid
    ax3a.axhline(y=0, c='white')
    ax3a.axvline(x=0, c='white')

    h2o_stat = summary_table(hws_obs['Qle'], spa_day['lemod'])
    co2_stat = summary_table(hws_obs['GPPgC'], spa_day['GPPgC'])

    sumtxt = lambda x: [dlab.upper() + ' = ' + "%.2f" % (ddat) for (dlab, ddat) in x.iteritems()]
    ax3a.text(0.25, 0.75, "\n".join(sumtxt(h2o_stat)), transform=ax3a.transAxes, \
              verticalalignment='center', horizontalalignment='center', color='blue')
    ax3a.text(0.75, 0.25, "\n".join(sumtxt(co2_stat)), transform=ax3a.transAxes, \
              verticalalignment='center', horizontalalignment='center', color='red')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.07)
    plt.savefig(SAVEPATH)
    #plt.show()

if __name__ == "__main__":

    DATPATH = os.path.expanduser("~/Savanna/Data/HowardSprings_IAV/pickled/daily/")
    FILEPATH = os.path.expanduser("~/Savanna/Models/SPA1/outputs/site_co2/HS_Exp1/outputs/")

    DATFILE = DATPATH + "daily_tower_fluxes.pkl"
    SPAFILE = FILEPATH + "hourly.csv"

    SAVEPATH = os.path.expanduser("~/Savanna/Analysis/figures/IAV/validated_exp1.pdf")

    main()

