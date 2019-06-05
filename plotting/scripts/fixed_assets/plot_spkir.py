#!/usr/bin/env python
"""
Created on May 31 2019

@author: Lori Garzio
@brief: This script is used create plots of SPKIR data by deployment, month and delivery method. The user has the option
of selecting a specific time range to plot and only plotting data from the preferred method/stream.
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
                    try:
                        rms = '-'.join((r, row[ii]))
                    except TypeError:
                        continue
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

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)
            sci_vars = cf.return_science_vars(stream)
            print('\nPlotting {} {}'.format(r, deployment))
            array = subsite[0:2]
            filename = '_'.join(fname.split('_')[:-1])
            save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots')
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            # -------- plot entire deployment --------

            for var in sci_vars:
                print(var)
                vv = ds[var]
                fv = vv._FillValue
                v = vv.values.T  # transpose 2D array
                n_nan = np.sum(np.isnan(v))

                # convert fill values to nans
                v[v == fv] = np.nan
                n_fv = np.sum(np.isnan(v)) - n_nan

                # plot before global ranges are removed
                fig, ax = pf.plot_spkir(tm, v, vv.name, vv.units)
                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                sfile = '-'.join((filename, var, t0[:10]))
                pf.save_fig(save_dir, sfile)

                # reject data outside of global ranges
                [g_min, g_max] = cf.get_global_ranges(r, var)
                if g_min is not None and g_max is not None:
                    v[v < g_min] = np.nan
                    v[v > g_max] = np.nan
                    n_grange = np.sum(np.isnan(v)) - n_fv - n_nan
                else:
                    n_grange = 'no global ranges'

                # plot after global ranges are removed
                fig, ax = pf.plot_spkir(tm, v, vv.name, vv.units)
                title2 = 'removed: {} global ranges [{}, {}]'.format(n_grange, g_min, g_max)
                ax.set_title((title + '\n' + t0 + ' - ' + t1 + '\n' + title2), fontsize=9)
                sfile = '-'.join((filename, var, t0[:10], 'rmgr'))
                pf.save_fig(save_dir, sfile)

            # -------- break the deployment into months and plot --------

            save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', 'monthly')
            cf.create_dir(save_dir)

            # create list of start and end dates
            dt_start = dt.datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S')
            dt_end = dt.datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S')
            start_dates = [dt_start.strftime('%m-%d-%YT00:00:00')]
            end_dates = []
            ts1 = dt_start
            while ts1 <= dt_end:
                ts2 = ts1 + dt.timedelta(days=1)
                if ts2.month != ts1.month:
                    start_dates.append(ts2.strftime('%m-%d-%YT00:00:00'))
                    end_dates.append(ts1.strftime('%m-%d-%YT23:59:59'))
                ts1 = ts2
            end_dates.append(dt_end.strftime('%m-%d-%YT23:59:59'))

            for sd, ed in zip(start_dates, end_dates):
                sd_format = dt.datetime.strptime(sd, '%m-%d-%YT%H:%M:%S')
                ed_format = dt.datetime.strptime(ed, '%m-%d-%YT%H:%M:%S')
                ds_month = ds.sel(time=slice(sd_format, ed_format))
                if len(ds_month['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(sd, ed))
                    continue
                tm = ds_month['time'].values
                t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')

                for var in sci_vars:
                    print(var)
                    vv = ds_month[var]
                    fv = vv._FillValue
                    v = vv.values.T  # transpose 2D array
                    n_nan = np.sum(np.isnan(v))

                    # convert fill values to nans
                    v[v == fv] = np.nan
                    n_fv = np.sum(np.isnan(v)) - n_nan

                    # reject data outside of global ranges
                    [g_min, g_max] = cf.get_global_ranges(r, var)
                    if g_min is not None and g_max is not None:
                        v[v < g_min] = np.nan
                        v[v > g_max] = np.nan
                        n_grange = np.sum(np.isnan(v)) - n_fv - n_nan
                    else:
                        n_grange = 'no global ranges'

                    # plot after global ranges are removed
                    fig, ax = pf.plot_spkir(tm, v, vv.name, vv.units)
                    title2 = 'removed: {} global ranges [{}, {}]'.format(n_grange, g_min, g_max)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1 + '\n' + title2), fontsize=9)
                    sfile = '-'.join((filename, var, t0[:7], 'rmgr'))
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190530T175841-CE01ISSM-RID16-08-SPKIRB000-recovered_host-spkir_abj_dcl_instrument_recovered/catalog.html']
    start_time = None  # dt.datetime(2014, 4, 17, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2014, 5, 17, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
