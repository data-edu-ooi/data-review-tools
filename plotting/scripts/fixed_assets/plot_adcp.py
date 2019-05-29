#!/usr/bin/env python
"""
Created on May 29 2019

@author: Lori Garzio
@brief: This script is used create plots of science variables for ADCPs by deployment and delivery method.
The user has the option of selecting a specific time range to plot and only plotting data from the preferred
method/stream.
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import itertools
import matplotlib.pyplot as plt
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
            #ds_vars = list(ds.data_vars.keys()) + [x for x in ds.coords.keys() if 'pressure' in x]  # get pressure variable from coordinates

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
            save_dir = os.path.join(sDir, array, subsite, refdes, 'plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            for var in sci_vars:
                print(var)
                if var != 'bin_depths':
                    v = ds[var]
                    fv = v._FillValue

                    if len(v.dims) == 1:
                        # Check if the array is all NaNs
                        if sum(np.isnan(v.values)) == len(v.values):
                            print('Array of all NaNs - skipping plot.')

                        # Check if the array is all fill values
                        elif len(v[v != fv]) == 0:
                            print('Array of all fill values - skipping plot.')

                        else:
                            # reject fill values
                            ind = v.values != fv
                            t = tm[ind]
                            v = v[ind]

                            # Plot all data
                            fig, ax = pf.plot_timeseries(t, v, stdev=None)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            sfile = '-'.join((filename, v.name, t0[:10]))
                            pf.save_fig(save_dir, sfile)

                            # Plot data with outliers removed
                            fig, ax = pf.plot_timeseries(t, v, stdev=5)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            sfile = '-'.join((filename, v.name, t0[:10])) + '_rmoutliers'
                            pf.save_fig(save_dir, sfile)

                    else:
                        v = v.values.T.astype(float)

                        #convert -9999 and fill values to nans
                        v[v == fv] = np.nan
                        v[v == -9999] = np.nan

                        # remove data outside of 5 standard deviations
                        stdev = np.nanstd(v)
                        ul = np.nanmean(v) + stdev * 5
                        ll = np.nanmean(v) - stdev * 5
                        v[v < ll] = np.nan
                        v[v > ul] = np.nan

                        clabel = '{} ({})'.format(var, ds[var].units)
                        ylabel = 'bin_depths ({})'.format(ds['bin_depths'].units)

                        if 'echo' in var:
                            color = 'BuGn'
                        else:
                            color = 'RdBu'

                        fig, ax = plt.subplots()
                        pcm = ax.pcolormesh(tm, ds['bin_depths'].values.T, v, cmap=color)
                        plt.gca().set_ylim(bottom=-1)
                        ax.invert_yaxis()

                        bar = fig.colorbar(pcm, ax=ax, label=clabel, extend='both')
                        bar.formatter.set_useOffset(False)
                        bar.ax.tick_params(labelsize=8)

                        ax.set_ylabel(ylabel, fontsize=9)
                        pf.format_date_axis(ax, fig)

                        outliers = 'removed outliers +/- 5 SD'

                        ax.set_title((title + '\n' + t0 + ' - ' + t1 + '\n' + outliers), fontsize=9)
                        sfile = '-'.join((filename, var, t0[:10]))
                        pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180115-CE01ISSM-MFD35-04-ADCPTM000-recovered_inst-adcp_velocity_earth/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
