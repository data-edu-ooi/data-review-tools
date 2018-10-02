#! /usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
import numpy as np
import re


def format_date_axis(axis, figure):
    df = mdates.DateFormatter('%Y-%m-%d')
    axis.xaxis.set_major_formatter(df)
    figure.autofmt_xdate()


def plot_timeseries(x, y, stdev=None):
    if stdev is None:
        yD = y.data
        leg_text = ()
    else:
        ind = reject_outliers(y, stdev)
        yD = y.data[ind]
        x = x[ind]
        outliers = str(len(ind.data) - sum(ind.data))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(x, yD, '.', markersize=2)
    ax.set_ylabel((y.name + " (" + y.units + ")"), fontsize=9)
    format_date_axis(ax, fig)
    ax.legend(leg_text, loc='best', fontsize=6)
    return fig, ax
    

def reject_outliers(data, m=3):
    # function to reject outliers beyond 3 standard deviations of the mean.
    # data: numpy array containing data
    # m: the number of standard deviations from the mean. Default: 3
    return abs(data - np.nanmean(data)) < m * np.nanstd(data)
    

def save_fig(save_dir, file_name, res=300):
    # save figure to a directory with a resolution of 300 DPI
    save_file = os.path.join(save_dir, file_name)
    plt.savefig(str(save_file), dpi=res)
    plt.close()


def science_vars(ds_variables):
    # return a list of only science variables
    misc_vars = ['quality', 'string', 'timestamp', 'deployment', 'provenance', 'qc', 'time', 'mission', 'obs', 'id',
                 'serial_number', 'volt', 'ref', 'sig', 'amp', 'rph', 'calphase', 'phase', 'therm', 'description']
    reg_ex = re.compile('|'.join(misc_vars))
    sci_vars = [s for s in ds_variables if not reg_ex.search(s)]
    return sci_vars


def y_axis_disable_offset(define_axis):
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    define_axis.yaxis.set_major_formatter(y_formatter)