#!/usr/bin/env python
"""
Created on June 13 2019

@author: Lori Garzio
@brief: This script is used create plots of PRESF data by deployment and delivery method. The user has the option of
selecting a specific time range to plot and only plotting data from the preferred method/stream.
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf


def main(sDir, url_list, start_time, end_time, preferred_only):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list and 'PRESF' in rd:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                for ud in udatasets:  # filter out collocated data files
                    if 'PRESF' in ud.split('/')[-1]:
                        datasets.append(ud)
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
            save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            for var in sci_vars:
                print(var)
                if var != 'id':
                #if var == 'presf_wave_burst_pressure':
                    y = ds[var]
                    fv = y._FillValue
                    if len(y.dims) == 1:

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
                            fig, ax = pf.plot_timeseries(t, y, y.name, stdev=None)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            sfile = '-'.join((filename, y.name, t0[:10]))
                            pf.save_fig(save_dir, sfile)

                            # Plot data with outliers removed
                            fig, ax = pf.plot_timeseries(t, y, y.name, stdev=5)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            sfile = '-'.join((filename, y.name, t0[:10])) + '_rmoutliers'
                            pf.save_fig(save_dir, sfile)
                    else:
                        v = y.values.T
                        n_nan = np.sum(np.isnan(v))

                        # convert fill values to nans
                        try:
                            v[v == fv] = np.nan
                        except ValueError:
                            v = v.astype(float)
                            v[v == fv] = np.nan
                        n_fv = np.sum(np.isnan(v)) - n_nan

                        # plot before global ranges are removed
                        fig, ax = pf.plot_presf_2d(tm, v, y.name, y.units)
                        ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                        sfile = '-'.join((filename, var, t0[:10]))
                        pf.save_fig(save_dir, sfile)

                        # reject data outside of global ranges
                        [g_min, g_max] = cf.get_global_ranges(r, var)
                        if g_min is not None and g_max is not None:
                            v[v < g_min] = np.nan
                            v[v > g_max] = np.nan
                            n_grange = np.sum(np.isnan(v)) - n_fv - n_nan

                            if n_grange > 0:
                                # don't plot if the array is all nans
                                if len(np.unique(np.isnan(v))) == 1 and np.unique(np.isnan(v))[0] == True:
                                    continue
                                else:
                                    # plot after global ranges are removed
                                    fig, ax = pf.plot_presf_2d(tm, v, y.name, y.units)
                                    title2 = 'removed: {} global ranges [{}, {}]'.format(n_grange, g_min, g_max)
                                    ax.set_title((title + '\n' + t0 + ' - ' + t1 + '\n' + title2), fontsize=9)
                                    sfile = '-'.join((filename, var, t0[:10], 'rmgr'))
                                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190607T172311-CP01CNSM-MFD35-02-PRESFB000-recovered_inst-presf_abc_wave_burst_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190607T172311-CP01CNSM-MFD35-02-PRESFB000-recovered_inst-presf_abc_tide_measurement_recovered/catalog.html']
    start_time = None  # dt.datetime(2015, 6, 3, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2019, 1, 1, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
