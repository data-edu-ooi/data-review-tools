#! /usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
import numpy as np
import functions.common as cf
import matplotlib.cm as cm
import pandas as pd  # We need pandas for this

def get_units(variable):
    try:
        var_units = variable.units
    except AttributeError:
        var_units = 'no_units'

    return var_units


def format_date_axis(axis, figure):
    df = mdates.DateFormatter('%Y-%m-%d')
    axis.xaxis.set_major_formatter(df)
    figure.autofmt_xdate()


def plot_profiles(x, y, t, ylabel, xlabel, clabel, end_times, deployments, stdev=None):
    """
    Create a profile plot for mobile instruments
    :param x: .nc data array containing data for plotting variable of interest (e.g. density)
    :param y: .nc data array containing data for plotting on the y-axis (e.g. pressure)
    :param t: .nc data array containing time data to be used for coloring (x,y) data pairs
    :param stdev: desired standard deviation to exclude from plotting
    """
    if type(t) is not np.ndarray:
        t = t.values

    if type(y) is not np.ndarray:
        y = y.values

    if type(x) is not np.ndarray:
        x = x.values

    if stdev is None:
        xD = x
        yD = y
        tD = t
        leg_text = ()
    else:
        ind2 = cf.reject_outliers(x, stdev)
        xD = x[ind2]
        yD = y[ind2]
        tD = t[ind2]
        outliers = str(len(x) - len(xD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.margins(y=.08, x=.02)
    plt.grid()
    sct = ax.scatter(xD, yD, c=tD, s=2, edgecolor='None', cmap='rainbow')
    cbar = plt.colorbar(sct, label=clabel)
    # cbar.ax.set_yticklabels(pd.to_datetime(end_times).strftime(date_format='%Y-%m-%d'), update_ticks=True)
    cbar.ax.set_yticklabels(pd.to_datetime(cbar.get_ticks()).strftime(date_format='%Y-%m-%d'))

    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(leg_text, loc='best', fontsize=6)

    return fig, ax


def plot_timeseries_all(x, y, y_name, y_units, stdev=None):
    """
    Create a simple timeseries plot
    :param x: array containing data for x-axis (e.g. time)
    :param y: array containing data for y-axis
    :param stdev: desired standard deviation to exclude from plotting
    """
    if stdev is None:
        xD = x
        yD = y
        leg_text = ()
    else:
        ind = cf.reject_extreme_values(y)
        ydata = y[ind]
        xdata = x[ind]

        ind2 = cf.reject_outliers(ydata, stdev)
        yD = ydata[ind2]
        xD = xdata[ind2]


        # ind2 = cf.reject_outliers(y, stdev)
        # yD = y[ind2]
        # xD = x[ind2]
        outliers = str(len(y) - len(yD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(xD, yD, '.', markersize=2)

    ax.set_ylabel((y_name + " (" + y_units + ")"), fontsize=9)
    format_date_axis(ax, fig)
    y_axis_disable_offset(ax)
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
        xD = x
        yD = y.values
        leg_text = ()
    else:
        ind = cf.reject_extreme_values(y.values)
        ydata = y[ind]
        xdata = x[ind]

        ind2 = cf.reject_outliers(ydata.values, stdev)
        yD = ydata[ind2].values
        xD = xdata[ind2]
        outliers = str(len(y) - len(yD))
        leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

    fig, ax = plt.subplots()
    plt.grid()
    plt.plot(xD, yD, '.', markersize=2)

    y_units = get_units(y)

    ax.set_ylabel((y.name + " (" + y_units + ")"), fontsize=9)
    format_date_axis(ax, fig)
    y_axis_disable_offset(ax)
    ax.legend(leg_text, loc='best', fontsize=6)
    return fig, ax


def plot_timeseries_compare(t0, t1, var0, var1, m0, m1, long_name, stdev=None):
    """
    Create a timeseries plot containing two datasets
    :param t0: data array of time for dataset 0
    :param t1: data array of time for dataset 1
    :param var0: .nc data array for plotting on the y-axis for dataset 0, including data values and variable attributes
    :param var1: .nc data array for plotting on the y-axis for dataset 1, including data values and variable attributes
    :param stdev: desired standard deviation to exclude from plotting
    """
    if stdev is None:
        t0_data = t0.values
        var0_data = var0.values
        leg_text = ('{}'.format(m0),)
        t1_data = t1.values
        var1_data = var1.values
        leg_text += ('{}'.format(m1),)
    else:
        ind0 = cf.reject_extreme_values(var0.values)
        t0i = t0[ind0]
        var0i = var0[ind0]

        ind02 = cf.reject_outliers(var0i.values, stdev)
        t0_data = t0i[ind02].values
        var0_data = var0i[ind02].values
        var0_data[var0_data <= 0.0] = np.nan  # get rid of zeros and negative numbers
        outliers0 = str((len(var0) - len(var0_data)) + (len(t0_data) - np.count_nonzero(~np.isnan(var0_data))))
        leg_text = ('{}: removed {} outliers (SD={})'.format(m0, outliers0, stdev),)

        ind1 = cf.reject_extreme_values(var1.values)
        t1i = t1[ind1]
        var1i = var1[ind1]

        ind12 = cf.reject_outliers(var1i.values, stdev)
        t1_data = t1i[ind12].values
        var1_data = var1i[ind12].values
        var1_data[var1_data <= 0.0] = np.nan  # get rid of zeros and negative numbers
        outliers1 = str((len(var1) - len(var1_data)) + (len(t1_data) - np.count_nonzero(~np.isnan(var1_data))))
        leg_text += ('{}: removed {} outliers (SD={})'.format(m1, outliers1, stdev),)

    y_units = get_units(var0)

    fig, ax = plt.subplots()
    plt.grid()
    #plt.ylim([2000, 2500])

    ax.plot(t0_data, var0_data, 'o', markerfacecolor='none', markeredgecolor='r', markersize=5, lw=.75)
    #ax.plot(t1_data, var1_data, 'x', markeredgecolor='b', markersize=5, lw=.75)
    ax.plot(t1_data, var1_data, '.', markeredgecolor='b', markersize=2)
    ax.set_ylabel((long_name + " (" + y_units + ")"), fontsize=9)
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
            yD = y.values
            xD = x
            leg_text = ()
        else:
            ind = cf.reject_extreme_values(y.values)
            ydata = y[ind]
            xdata = x[ind]

            ind2 = cf.reject_outliers(ydata.values, stdev)
            yD = ydata[ind2].values
            xD = xdata[ind2]
            outliers = str(len(y) - len(yD))
            leg_text = ('{}: rm {} outliers'.format(vars[i], outliers),)

        y_units = get_units(y)
        c = colors[i]
        ax[i].plot(xD, yD, '.', markersize=2, color=c)
        ax[i].set_ylabel(('(' + y_units + ')'), fontsize=5)
        ax[i].tick_params(axis='y', labelsize=6)
        ax[i].legend(leg_text, loc='best', fontsize=4)
        y_axis_disable_offset(ax[i])
        if i == len(vars) - 1:  # if the last variable has been plotted
            format_date_axis(ax[i], fig)

    return fig, ax


def plot_ts(sal_vector, temp_vector, dens, salinity, temperature, cbar):
    fig, ax = plt.subplots()
    CS = plt.contour(sal_vector, temp_vector, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%1.0f')  # Label every second level
    xc = ax.scatter(salinity, temperature, c=cbar, s=2, edgecolor='None')

    # add colorbar
    #bar = fig.colorbar(xc, ax=ax)
    #bar
    #bar.formatter.set_useOffset(False)

    ax.set_xlabel('Salinity')
    ax.set_ylabel('Temperature (C)')

    return fig, ax


def plot_xsection(subsite, x, y, z, clabel, ylabel, stdev=None):
    """
    Create a cross-section plot for mobile instruments
    :param subsite: subsite part of reference designator to plot
    :param x:  array containing data for x-axis (e.g. time)
    :param y: .nc data array containing data for plotting on the y-axis (e.g. pressure)
    :param z: .nc data array containing data for plotting variable of interest (e.g. density)
    :param stdev: desired standard deviation to exclude from plotting
    """
    if type(z) is not np.ndarray:
        z = z.values

    if type(y) is not np.ndarray:
        y = y.values

    if type(x) is not np.ndarray:
        x = x.values

    # when plotting gliders, remove zeros (glider fill values) and negative numbers
    if 'MOAS' in subsite:
        z[z <= 0.0] = np.nan
        zeros = str(len(z) - np.count_nonzero(~np.isnan(z)))

    if stdev is None:
        xD = x
        yD = y
        zD = z
    else:
        ind = cf.reject_extreme_values(z)
        xdata = x[ind]
        ydata = y[ind]
        zdata = z[ind]
        
        ind2 = cf.reject_outliers(zdata, stdev)
        xD = xdata[ind2]
        yD = ydata[ind2]
        zD = zdata[ind2]
        outliers = str(len(zdata) - len(zD))

    try:
        zeros
    except NameError:
        zeros = None

    try:
        outliers
    except NameError:
        outliers = None

    fig, ax = plt.subplots()
    plt.margins(y=.08, x=.02)
    xc = ax.scatter(xD, yD, c=zD, s=2, edgecolor='None')
    ax.invert_yaxis()

    # add color bar
    bar = fig.colorbar(xc, ax=ax, label=clabel)
    bar.formatter.set_useOffset(False)

    ax.set_ylabel(ylabel, fontsize=9)
    format_date_axis(ax, fig)

    if zeros is None and type(outliers) is str:
        leg = ('rm: {} outliers (SD={})'.format(outliers, stdev),)
        ax.legend(leg, loc=1, fontsize=6)
    if type(zeros) is str and outliers is None:
        leg = ('rm: {} values <=0.0'.format(zeros),)
        ax.legend(leg, loc=1, fontsize=6)
    if type(zeros) is str and type(outliers) is str:
        leg = ('rm: {} values <=0.0, rm: {} outliers (SD={})'.format(zeros, outliers, stdev),)
        ax.legend(leg, loc=1, fontsize=6)
    return fig, ax


def pressure_var(dataset, vars):
    """
    Return the pressure (dbar) variable in a dataset.
    :param vars: list of all variables in a dataset
    """
    pressure_variables = ['int_ctd_pressure', 'seawater_pressure', 'ctdpf_ckl_seawater_pressure', 'sci_water_pressure_dbar',
                     'ctdbp_seawater_pressure', 'ctdmo_seawater_pressure', 'ctdbp_no_seawater_pressure',
                     'sci_water_pressure_dbar', 'pressure_depth', 'abs_seafloor_pressure', 'presf_tide_pressure',
                     'presf_wave_burst_pressure', 'pressure', 'velpt_pressure', 'ctd_dbar', 'vel3d_k_pressure',
                     'seafloor_pressure', 'pressure_mbar']
    pvariables = list(set(pressure_variables).intersection(vars))
    if pvariables:
        pvars = []
        for press_var in pvariables:
            if press_var == 'int_ctd_pressure':
                pvars.append(str(press_var))
            else:
                try:
                    units = dataset[press_var].units
                    if units in ['dbar', '0.001 dbar']:
                        pvars.append(str(press_var))
                except AttributeError:
                    continue

        if len(pvars) > 1:
            print('More than 1 pressure variable found in the file')
        elif len(pvars) == 1:
            pvar = str(pvars[0])
        else:
            pvar = None
    else:
        y = [x for x in list(dataset.coords.keys()) if 'pressure' in x]
        if len(y) > 0:
            pvar = str(y[0])
        else:
            pvar = None
    return pvar


def save_fig(save_dir, file_name, res=150):
    # save figure to a directory with a resolution of 150 DPI
    save_file = os.path.join(save_dir, file_name)
    plt.savefig(str(save_file), dpi=res)
    plt.close()


def y_axis_disable_offset(axis):
    # format y-axis to disable offset
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    axis.yaxis.set_major_formatter(y_formatter)
