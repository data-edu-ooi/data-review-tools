#!/usr/bin/env python
"""
Created on Dec 12 2019

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for a reference designator by
deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
The user has the option of selecting a specific time range to plot. This is for cabled HPIES data because .nc files
need to be downloaded from THREDDS
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import itertools
import functions.common as cf
import functions.plotting as pf


def main(sDir, ncdir, start_time, end_time):
    rd_list = [ncdir.split('/')[-2]]

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for root, dirs, files in os.walk(ncdir):
            for f in files:
                if f.endswith('.nc'):
                    datasets.append(f)
        # for u in url_list:
        #     splitter = u.split('/')[-2].split('-')
        #     rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
        #     if rd_check == r:
        #         udatasets = cf.get_nc_urls([u])
        #         datasets.append(udatasets)
        #datasets = list(itertools.chain(*datasets))
        for fd in datasets:
            if '_blank' not in fd:
                ds = xr.open_dataset(os.path.join(ncdir, fd), mask_and_scale=False)
                ds = ds.swap_dims({'obs': 'time'})
                ds_vars = list(ds.data_vars.keys()) + [x for x in ds.coords.keys() if 'pressure' in x]  # get pressure variable from coordinates
                #raw_vars = cf.return_raw_vars(ds_vars)

                if start_time is not None and end_time is not None:
                    ds = ds.sel(time=slice(start_time, end_time))
                    if len(ds['time'].values) == 0:
                        print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                        continue

                fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(os.path.join(ncdir, fd))
                if 'NUTNR' in refdes or 'VEL3D in refdes':
                    vars = cf.return_science_vars(stream)
                else:
                    vars = cf.return_raw_vars(ds_vars)
                print('\nPlotting {} {}'.format(r, deployment))
                array = subsite[0:2]
                filename = '_'.join(fname.split('_')[:-1])
                save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', deployment)
                cf.create_dir(save_dir)

                tm = ds['time'].values
                t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, refdes, method))

                for var in vars:
                    print(var)
                    if var not in ['id', 'record_type', 'unique_id']:  # if var != 'id'
                        y = ds[var]
                        try:
                            fv = y._FillValue
                        except AttributeError:
                            fv = np.nan
                        if len(y.dims) == 1:
                            # Check if the array is all NaNs
                            y[y == fv] = np.nan  # turn fill values to nans
                            if sum(np.isnan(y.values)) == len(y.values):
                                print('Array of all NaNs and/or fill values - skipping plot.')

                            # Check if the array is all fill values
                            # elif len(y[y != fv]) == 0:
                            #     print('Array of all fill values - skipping plot.')

                            else:
                                # reject fill values
                                ind = y.values != fv
                                t = tm[ind]
                                y = y[ind]

                                # Plot all data
                                fig, ax = pf.plot_timeseries(t, y, y.name, stdev=None)
                                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                                sfile = '-'.join((filename, y.name, t0[:10]))
                                pf.save_fig(save_dir, sfile)

                                # Plot data with outliers removed
                                fig, ax = pf.plot_timeseries(t, y, y.name, stdev=5)
                                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                                sfile = '-'.join((filename, y.name, t0[:10])) + '_rmoutliers'
                                pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    ncdir = '/Users/lgarzio/Documents/OOI/DataReviews/RS/RS03AXBS/RS03AXBS-LJ03A-05-HPIESA301/data'
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    main(sDir, ncdir, start_time, end_time)
