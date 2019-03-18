#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for a reference designator by
deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
The user has the option of selecting a specific time range to plot and only plotting data from the preferred
method/stream.
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import itertools
import functions.common as cf
import functions.plotting as pf


def main(sDir, url_list, start_time, end_time, preferred_only):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        fdatasets = []
        if preferred_only == 'yes':
            # get the preferred stream information
            ps_df, n_streams = cf.get_preferred_stream_info(r)
            for index, row in ps_df.iterrows():
                for ii in range(n_streams):
                    rms = '-'.join((r, row[ii]))
                    for dd in datasets:
                        spl = dd.split('/')[-2].split('-')
                        catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                        fdeploy = dd.split('/')[-1].split('_')[0]
                        if rms == catalog_rms and fdeploy == row['deployment']:
                            fdatasets.append(dd)
        else:
            fdatasets = datasets

        fdatasets = np.unique(fdatasets).tolist()
        for fd in fdatasets:
            ds = xr.open_dataset(fd, mask_and_scale=False)
            ds = ds.swap_dims({'obs': 'time'})
            ds_vars = list(ds.data_vars.keys()) + [x for x in ds.coords.keys() if 'pressure' in x]  # get pressure variable from coordinates
            raw_vars = cf.return_raw_vars(ds_vars)

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)
            print('\nPlotting {} {}'.format(r, deployment))
            array = subsite[0:2]
            filename = '_'.join(fname.split('_')[:-1])
            save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            for var in raw_vars:
                print(var)
                y = ds[var]
                fv = y._FillValue

                # Check if the array is all NaNs
                if sum(np.isnan(y.values)) == len(y.values):
                    print('Array of all NaNs - skipping plot.')

                # Check if the array is all fill values
                elif len(y[y != fv]) == 0:
                    print('Array of all fill values - skipping plot.')

                else:
                    # reject fill values
                    ind = y.values != fv
                    t = tm[ind]
                    y = y[ind]

                    # Plot all data
                    fig, ax = pf.plot_timeseries(t, y, stdev=None)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '-'.join((filename, y.name, t0[:10]))
                    pf.save_fig(save_dir, sfile)

                    # Plot data with outliers removed
                    fig, ax = pf.plot_timeseries(t, y, stdev=5)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '-'.join((filename, y.name, t0[:10])) + '_rmoutliers'
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172034-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172050-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172104-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
