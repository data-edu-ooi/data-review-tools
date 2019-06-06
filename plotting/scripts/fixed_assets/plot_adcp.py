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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf

def dropna(arr, *args, **kwarg):
    # turn 2D numpy array into a data frame and drop nan
    assert isinstance(arr, np.ndarray)
    dropped=pd.DataFrame(arr).dropna(*args, **kwarg)

    if arr.ndim==1:
        dropped=dropped.values.flatten()
    return dropped

def in_list(x, ix):
    # keep listed entries with specific words.
    y = [el for el in x if any(ignore in el for ignore in ix)]
    return y

def notin_list(x, ix):
    # filter out list entries with specific words.
    y = [el for el in x if not any(ignore in el for ignore in ix)]
    return y

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
            sci_vars = notin_list(sci_vars, ['bin_depths', 'salinity', 'temperature', 'beam'])
            sci_vars = [name for name in sci_vars if ds[name].units != 'mm s-1']


            print('\nPlotting {} {}'.format(r, deployment))
            array = subsite[0:2]
            filename = '_'.join(fname.split('_')[:-1])
            save_dir = os.path.join(sDir, array, subsite, refdes, 'plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title_text = ' '.join((deployment, refdes, method))

            for var in sci_vars:
                print(var)
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
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=9)
                        sfile = '-'.join((filename, v.name, t0[:10]))
                        pf.save_fig(save_dir, sfile)

                        # Plot data with outliers removed
                        fig, ax = pf.plot_timeseries(t, v, stdev=5)
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=9)
                        sfile = '-'.join((filename, v.name, t0[:10])) + '_rmoutliers'
                        pf.save_fig(save_dir, sfile)

                else:
                    v = v.values.T.astype(float)
                    n_nan = np.sum(np.isnan(v))

                    #convert -9999 and fill values to nans
                    v[v == fv] = np.nan
                    v[v == -9999] = np.nan
                    n_fv = np.sum(np.isnan(v)) - n_nan

                    # reject data outside of global ranges
                    [g_min, g_max] = cf.get_global_ranges(r, var)
                    if g_min is not None and g_max is not None:
                        v[v < g_min] = np.nan
                        v[v > g_max] = np.nan
                        n_grange = np.sum(np.isnan(v)) - n_fv - n_nan
                    else:
                        n_grange = 'no global ranges'

                    ylabel = 'bin_depths ({})'.format(ds['bin_depths'].units)
                    clabel = '{} ({})'.format(var, ds[var].units)
                    y = ds['bin_depths'].values.T
                    y_nan = np.sum(np.isnan(y))

                    # remove extreme bin_depths
                    y = np.where(y < 6000, y, np.nan)
                    # y[y > 6000] = np.nan
                    bin_nan = np.sum(np.isnan(y)) - y_nan

                    bin_title = 'removed: {} bin depths > 6000'.format(bin_nan)

                    if 'echo' in var:
                        color = 'BuGn'
                    else:
                        color = 'RdBu'

                    new_y = dropna(y, axis=1)
                    y_mask = new_y.loc[list(new_y.index), list(new_y.columns)]
                    v_new = pd.DataFrame(v)
                    v_mask = v_new.loc[list(new_y.index), list(new_y.columns)]
                    tm_mask = tm[new_y.columns]

                    fig, ax, __ = pf.plot_adcp(tm_mask, np.array(y_mask), np.array(v_mask), ylabel, clabel, color)
                    if bin_nan > 0:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + bin_title), fontsize=9)
                    else:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=9)
                    # ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '-'.join((filename, var, t0[:10]))
                    pf.save_fig(save_dir, sfile)

                    fig, ax, n_nans_all = pf.plot_adcp(tm_mask, np.array(y_mask), np.array(v_mask), ylabel, clabel, color, stdev=5)
                    if type(n_grange) == str:
                        outl = n_nans_all - n_fv - n_nan
                    else:
                        outl = n_nans_all - n_grange - n_fv - n_nan
                    title2 = 'removed: {} fill values, {} GR [{}, {}], {} outliers +/- 5 SD'.format(n_fv, n_grange,
                                                                                                    g_min, g_max, outl)
                    if bin_nan > 0:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title2 + '\n' + bin_title), fontsize=8)
                    else:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title2), fontsize=8)
                    # ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title2), fontsize=8)
                    sfile = '-'.join((filename, var, t0[:10])) + '_rmoutliers'
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list =['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180207-CE06ISSM-MFD35-04-ADCPTM000-telemetered-adcp_velocity_earth/catalog.html',
               'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180205-CE06ISSM-MFD35-04-ADCPTM000-recovered_inst-adcp_velocity_earth/catalog.html',
               '']
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T175913-CP01CNSM-MFD35-01-ADCPTF000-recovered_inst-adcp_velocity_earth/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
