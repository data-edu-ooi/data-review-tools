#! /usr/bin/env python
"""
Created on Oct 1 2018

@author: Lori Garzio
@brief: This script is used to plot a timeseries of all CTDMO data from an entire platform. Outputs two plots of each
science variable by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard
deviations.

@usage:
sDir: local directory to which plots are saved
f: csv file containing datasets from one platform to plot (e.g. output from one of the data download tools
ooi-data-lab/data-review-tools/data_download with a column labeled 'outputUrl'.
Example file 'ctdmo_data_request_summary.csv')
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import functions.common as cf
import functions.plotting as pf


def deploy_method_stream(ds):
    fname = ds.split('/')[-1].split('.nc')[0]
    deployment = fname[0:14]
    method = fname.split('-')[4]
    stream_comp = fname.split('-')[5].split('_')
    stream = '_'.join(stream_comp[:-1])
    dms = '-'.join((deployment, method, stream))
    return dms


def plot_ctdmo(data_dict, var, stdev=None):
    colors10 = ['red', 'firebrick', 'orange', 'mediumseagreen', 'blue', 'darkgreen', 'purple', 'indigo', 'slategray',
                'black']

    colors16 = ['red', 'firebrick', 'orange', 'gold', 'mediumseagreen', 'darkcyan', 'blue', 'darkgreen', 'purple',
                'lightgray', 'slategray', 'black', 'coral', 'gold', 'limegreen', 'midnightblue']

    fig, ax1 = plt.subplots()
    sensor_list = []
    median_list = []

    for i, (key, value) in enumerate(data_dict.items()):
        if len(data_dict) < 11:
            colors = colors10
        else:
            colors = colors16
        t = value['time']
        y = value['yD']
        if stdev != None:
            ind = cf.reject_outliers(value['yD'], stdev)
            t = t[ind]
            y = y[ind]

        refdes = str(key)
        sensor_list.append(refdes.split('-')[-1])
        median_list.append(value['median'])

        plt.scatter(t, y, c=colors[i], marker='.', s=.5)

        if i == len(data_dict) - 1:  # if the last dataset has been plotted
            plt.grid()
            plt.margins(y=.05, x=.05)

            # refdes on secondary y-axis only for pressure and density
            if var in ['ctdmo_seawater_pressure', 'density']:
                ax2 = ax1.twinx()
                ax2.set_ylim(ax1.get_ylim())
                plt.yticks(median_list, sensor_list, fontsize=7.5)
                plt.subplots_adjust(right=.85)

            pf.format_date_axis(ax1, fig)
            pf.y_axis_disable_offset(ax1)

            subsite = refdes.split('-')[0]
            title = subsite + ' ' + ('-'.join((value['dms'].split('-')[0], value['dms'].split('-')[1])))
            ax1.set_ylabel((var + " (" + value['yunits'] + ")"), fontsize=9)
            ax1.set_title(title, fontsize=10)

            fname = '-'.join((subsite, value['dms'], var))
            if stdev != None:
                fname = '-'.join((fname, 'outliers_rejected'))
            sdir = os.path.join(sDir, subsite, value['dms'].split('-')[0])
            cf.create_dir(sdir)
            pf.save_fig(sdir, fname)


def main(sDir, f):
    ff = pd.read_csv(os.path.join(sDir, f))
    datasets = cf.get_nc_urls(ff['outputUrl'].tolist())

    plt_vars = ['ctdmo_seawater_pressure', 'ctdmo_seawater_temperature', 'ctdmo_seawater_conductivity',
                'practical_salinity', 'density']

    dms_list = []
    for ds in datasets:
        dms = deploy_method_stream(ds)
        if dms not in dms_list:
            dms_list.append(dms)

    for dd in dms_list:
        for v in plt_vars:
            print(v)
            data = OrderedDict()
            for ds in datasets:
                dms = deploy_method_stream(ds)
                if dms == dd:
                    f = xr.open_dataset(ds)
                    f = f.swap_dims({'obs': 'time'})
                    refdes = '-'.join((f.subsite, f.node, f.sensor))
                    yD = f[v].values
                    data[refdes] = {}
                    data[refdes]['time'] = f['time'].values
                    data[refdes]['yD'] = yD
                    data[refdes]['yunits'] = f[v].units
                    data[refdes]['median'] = np.median(yD)
                    data[refdes]['dms'] = dms

            plot_ctdmo(data, v)
            plot_ctdmo(data, v, 5)  # reject outliers beyond 5 standard deviations


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary_20181001T1106.csv'
    main(sDir, f)
