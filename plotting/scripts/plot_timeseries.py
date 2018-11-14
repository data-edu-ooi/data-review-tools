#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for a reference designator by
deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
The user has the option of selecting a specific time range to plot.
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import functions.common as cf
import functions.plotting as pf


def main(sDir, f, start_time, end_time):
    ff = pd.read_csv(os.path.join(sDir, f))
    datasets = cf.get_nc_urls(ff['outputUrl'].tolist())
    for i, d in enumerate(datasets):
        print('\nDataset {} of {}: {}'.format(i + 1, len(datasets), d))
        with xr.open_dataset(d, mask_and_scale=False) as ds:
            ds = ds.swap_dims({'obs': 'time'})
            raw_vars = cf.return_raw_vars(ds.data_vars.keys())

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].data) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)
            save_dir = os.path.join(sDir, subsite, refdes, 'timeseries_plots', deployment)
            cf.create_dir(save_dir)

            t = ds['time'].data
            t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            for var in raw_vars:
                print(var)
                y = ds[var]
                fv = y._FillValue

                # Check if the array is all NaNs
                if sum(np.isnan(y.data)) == len(y.data):
                    print('Array of all NaNs - skipping plot.')

                # Check if the array is all fill values
                elif len(y[y != fv]) == 0:
                    print('Array of all fill values - skipping plot.')

                else:
                    # Plot all data
                    fig, ax = pf.plot_timeseries(t, y, stdev=None)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '_'.join((fname, y.name))
                    pf.save_fig(save_dir, sfile)

                    # Plot data with outliers removed
                    fig, ax = pf.plot_timeseries(t, y, stdev=5)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '_'.join((fname, y.name, 'rmoutliers'))
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary.csv'
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    main(sDir, f, start_time, end_time)
