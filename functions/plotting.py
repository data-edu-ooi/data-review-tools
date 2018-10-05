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


def plot_profiles(x, y, colors, stdev=None):
    """
    Create a profile plot for mobile instruments
    :param x: .nc data array containing data for plotting variable of interest (e.g. density)
    :param y: .nc data array containing data for plotting on the y-axis (e.g. pressure)
    :param colors: list of colors to be used for plotting
    :param stdev: desired standard deviation to exclude from plotting
    """
    if stdev is None:
        xD = x.data
        yD = y.data
        leg_text = ()
    else:
        ind = reject_outliers(x, stdev)
        xD = x.data[ind]
        yD = y.data[ind]
        outliers = str(len(x) - len(xD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.grid()
    ax.scatter(xD, yD, c=colors, s=2, edgecolor='None')
    ax.invert_yaxis()
    ax.set_xlabel((x.name + " (" + x.units + ")"), fontsize=9)
    ax.set_ylabel((y.name + " (" + y.units + ")"), fontsize=9)
    ax.legend(leg_text, loc='best', fontsize=6)
    return fig, ax


def plot_timeseries(x, y, stdev=None):
    """
    Create a simple timeseries plot
    :param x: array containing data for x-axis (e.g. time)
    :param y: .nc data array for plotting on the y-axis, including data values, coordinates, and variable attributes
    :param stdev: desired standard deviation to exclude from plotting
    """
    if stdev is None:
        yD = y.data
        leg_text = ()
    else:
        ind = reject_outliers(y, stdev)
        yD = y.data[ind]
        x = x[ind]
        outliers = str(len(y) - len(yD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(x, yD, '.', markersize=2)
    ax.set_ylabel((y.name + " (" + y.units + ")"), fontsize=9)
    format_date_axis(ax, fig)
    y_axis_disable_offset(ax)
    ax.legend(leg_text, loc='best', fontsize=6)
    return fig, ax

def plot_timeseries_panel(ds, x, vars, colors, stdev=None):
    """
    Create a timeseries plot with horizontal panels of each science parameter
    :param ds: dataset (e.g. .nc file opened with xarray) containing data for plotting
    :param x: array containing data for x-axis (e.g. time)
    :param vars: list of science variables to plot
    :param colors: list of colors to be used for plotting
    :param stdev: desired standard deviation to exclude from plotting
    """
    fig, ax = plt.subplots(len(vars), sharex=True)

    for i in range(len(vars)):
        y = ds[vars[i]]

        if stdev is None:
            yD = y.data
            xD = x
            leg_text = ()
        else:
            ind = reject_outliers(y, stdev)
            yD = y.data[ind]
            xD = x[ind]
            outliers = str(len(y) - len(yD))
            leg_text = ('{}: rm {} outliers'.format(y.name, outliers),)

        c = colors[i]
        ax[i].plot(xD, yD, '.', markersize=2, color=c)
        ax[i].set_ylabel(('(' + y.units + ')'), fontsize=5)
        ax[i].tick_params(axis='y', labelsize=6)
        ax[i].legend(leg_text, loc='best', fontsize=4)
        y_axis_disable_offset(ax[i])
        if i == len(vars) - 1:  # if the last variable has been plotted
            format_date_axis(ax[i], fig)
    return fig, ax


def plot_xsection(x, y, z, stdev=None):
    """
    Create a cross-section plot for mobile instruments
    :param x:  array containing data for x-axis (e.g. time)
    :param y: .nc data array containing data for plotting on the y-axis (e.g. pressure)
    :param z: .nc data array containing data for plotting variable of interest (e.g. density)
    :param stdev: desired standard deviation to exclude from plotting
    """
    if stdev is None:
        xD = x
        yD = y.data
        zD = z.data
        leg_text = ()
    else:
        ind = reject_outliers(z, stdev)
        xD = x[ind]
        yD = y.data[ind]
        zD = z.data[ind]
        outliers = str(len(z) - len(zD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    xc = ax.scatter(xD, yD, c=zD, s=2, edgecolor='None')
    ax.invert_yaxis()

    # add colorbar
    bar = fig.colorbar(xc, ax=ax, label=(z.name + " (" + z.units + ")"))
    bar
    bar.formatter.set_useOffset(False)

    ax.set_ylabel((y.name + " (" + y.units + ")"), fontsize=9)
    format_date_axis(ax, fig)
    ax.legend(leg_text, loc='best', fontsize=6)
    return fig, ax


def pressure_var(vars):
    """
    Return the pressure (dbar) variable in a dataset.
    :param vars: list of all variables in a dataset
    """
    pressure_vars = ['int_ctd_pressure', 'pressure', 'seawater_pressure', 'ctdpf_ckl_seawater_pressure',
                     'ctdbp_seawater_pressure', 'ctdmo_seawater_pressure', 'ctdbp_no_seawater_pressure',
                     'sci_water_pressure_dbar']
    pvars = list(set(pressure_vars).intersection(vars))
    if len(pvars) > 1:
        print('More than 1 pressure variable found in the file')
    elif len(pvars) == 0:
        print('No pressure variable found in the file')
    else:
        pvar = str(pvars[0])
        return pvar


def reject_outliers(data, m=3):
    """
    Reject outliers beyond m standard deviations of the mean.
    :param data: numpy array containing data
    :param m: number of standard deviations from the mean. Default: 3
    """
    return abs(data - np.nanmean(data)) < m * np.nanstd(data)
    

def save_fig(save_dir, file_name, res=150):
    # save figure to a directory with a resolution of 150 DPI
    save_file = os.path.join(save_dir, file_name)
    plt.savefig(str(save_file), dpi=res)
    plt.close()


def y_axis_disable_offset(axis):
    # format y-axis to disable offset
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    axis.yaxis.set_major_formatter(y_formatter)
