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


def reject_err_data_1_dims(y, y_fill, r, sv, n=None):
    n_nan = np.sum(np.isnan(y))  # count nans in data
    n_nan = n_nan.item()
    y = np.where(y != y_fill, y, np.nan)  # replace fill_values by nans in data
    y = np.where(y != -9999, y, np.nan)  # replace -9999 by nans in data
    n_fv = np.sum(np.isnan(y)) - n_nan  # re-count nans in data
    n_fv = n_fv.item()
    y = np.where(y > -1e10, y, np.nan)  # replace extreme values by nans in data
    y = np.where(y < 1e10, y, np.nan)
    n_ev = np.sum(np.isnan(y)) - n_fv - n_nan  # re-count nans in data
    n_ev = n_ev.item()

    g_min, g_max = cf.get_global_ranges(r, sv)  # get global ranges:
    if g_min and g_max:
        y = np.where(y >= g_min, y, np.nan)  # replace extreme values by nans in data
        y = np.where(y <= g_max, y, np.nan)
        n_grange = np.sum(np.isnan(y)) - n_ev - n_fv - n_nan  # re-count nans in data
        n_grange = n_grange.item()
    else:
        n_grange = np.nan

    stdev = np.nanstd(y)
    if stdev > 0.0:
        y = np.where(abs(y - np.nanmean(y)) < n * stdev, y, np.nan) # replace 5 STD by nans in data
        n_std = np.sum(np.isnan(y)) - n_grange - n_ev - n_fv - n_nan # re-count nans in data
        n_std = n_std.item()
    else:
        n_std = 0

    return y, n_nan, n_fv, n_ev, n_grange, g_min, g_max, n_std


def reject_err_data_2_dims(y, y_bad_beams, y_fill, r, sv):
    n_nan = np.sum(np.isnan(y))  # count nans in data
    y[y == y_fill] = np.nan  # replace fill_values by nans in data
    y[y == -9999] = np.nan  # replace -99999 by nans in data
    n_fv = np.sum(np.isnan(y)) - n_nan  # re-count nans in data
    y[y < -1e10] = np.nan  # replace extreme values by nans in data
    y[y > 1e10] = np.nan
    n_ev = np.sum(np.isnan(y)) - n_fv - n_nan  # re-count nans in data
    if type(y_bad_beams) == dict:  # if it's a dictionary, it's actually the percent of good beams
        for k in list(y_bad_beams.keys()):
            y[y_bad_beams[k] < 75] = np.nan
    else:
        y[y_bad_beams > 25] = np.nan  # replace bad beams by nans in data
    n_bb = np.sum(np.isnan(y)) - n_ev - n_fv - n_nan  # re-count nans in data

    [g_min, g_max] = cf.get_global_ranges(r, sv)  # get global ranges
    if g_min is not None and g_max is not None:
        y[y < g_min] = np.nan  # replace extreme values by nans in data
        y[y > g_max] = np.nan
        n_grange = np.sum(np.isnan(y)) - n_bb - n_ev - n_fv - n_nan  # re-count nans in data
    else:
        n_grange = np.nan

    return y, n_nan, n_fv, n_ev, n_bb, n_grange, g_min, g_max


def dropna(arr, *args, **kwarg):
    # turn 2D numpy array into a data frame and drop nan
    assert isinstance(arr, np.ndarray)
    dropped = pd.DataFrame(arr).dropna(*args, **kwarg)

    if arr.ndim == 1:
        dropped = dropped.values.flatten()
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
            # drop the following list of key words from science variables list
            sci_vars = notin_list(sci_vars, ['bin_depths', 'salinity', 'temperature'])
            sci_vars = [name for name in sci_vars if ds[name].units != 'mm s-1']

            print('\nPlotting {} {}'.format(r, deployment))
            array = subsite[0:2]
            filename = '_'.join(fname.split('_')[:-1])
            save_dir = os.path.join(sDir, array, subsite, refdes, 'preferred_method_plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title_text = ' '.join((deployment, refdes, method))

            try:
                ylabel = 'bin_depths ({})'.format(ds['bin_depths'].units)
            except KeyError:
                print('No bin_depth variable in file: cannot create plots')
                continue

            for var in sci_vars:
                print(var)
                v = ds[var]
                fv = v._FillValue
                v_name = v.long_name

                if len(v.dims) == 1:
                    v, n_nan, n_fv, n_ev, n_grange, g_min, g_max, n_std = reject_err_data_1_dims(v, fv, r, var, n=5)

                    if len(tm) > np.sum(np.isnan(v)):  # only plot if the array contains values
                        # Plot all data
                        fig, ax = pf.plot_timeseries(tm, v, v_name, stdev=None)
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=9)
                        sfile = '-'.join((filename, v_name, t0[:10]))
                        pf.save_fig(save_dir, sfile)

                        # Plot data with outliers removed
                        fig, ax = pf.plot_timeseries(tm, v, v_name, stdev=5)
                        title_i = 'removed: {} nans, {} fill values, {} extreme values, {} GR [{}, {}],' \
                                  ' {} outliers +/- 5 SD'.format(n_nan, n_fv , n_ev, n_grange, g_min, g_max, n_std)

                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title_i), fontsize=8)
                        sfile = '-'.join((filename, v_name, t0[:10])) + '_rmoutliers'
                        pf.save_fig(save_dir, sfile)
                    else:
                        print('Array of all nans - skipping plot')

                else:
                    v = v.values.T.astype(float)
                    try:
                        v_bad_beams = ds['percent_bad_beams']  # get bad beams percent
                        fv_bad_beam = v_bad_beams._FillValue
                        v_bad_beams = v_bad_beams.values.T.astype(float)
                        v_bad_beams[v_bad_beams == fv_bad_beam] = np.nan  # mask fill values
                    except KeyError:
                        print('No percent_bad_beams variable in file')
                        try:
                            # for cabled data, it's percent good beams
                            percentgood = {'percent_good_beam1': [], 'percent_good_beam2': [], 'percent_good_beam3': [], 'percent_good_beam4': []}
                            for pg in list(percentgood.keys()):
                                vv = ds[pg]
                                fv_vv = vv._FillValue
                                vv = vv.values.T.astype(float)
                                vv[vv == fv_vv] = np.nan
                                percentgood[pg] = vv
                            v_bad_beams = percentgood
                        except KeyError:
                            print('No percent_good_beams in file')
                            v_bad_beams = np.empty(np.shape(v))
                            v_bad_beams[:] = np.nan

                    v, n_nan, n_fv, n_ev, n_bb, n_grange, g_min, g_max = reject_err_data_2_dims(v, v_bad_beams, fv, r, var)

                    clabel = '{} ({})'.format(var, ds[var].units)

                    # check bin depths for extreme values
                    y = ds['bin_depths'].values.T
                    # if all the values are negative, take the absolute value (cabled data bin depths are negative)
                    if int(np.nanmin(y)) < 0 and int(np.nanmax(y)) < 0:
                        y = abs(y)
                    y_nan = np.sum(np.isnan(y))
                    y = np.where(y < 6000, y, np.nan)  # replace extreme bin_depths by nans
                    bin_nan = np.sum(np.isnan(y)) - y_nan
                    bin_title = 'removed: {} bin depths > 6000'.format(bin_nan)

                    if 'echo' in var:
                        color = 'BuGn'
                    else:
                        color = 'RdBu'

                    new_y = dropna(y, axis=1)  # convert to DataFrame to drop nan
                    y_mask = new_y.loc[list(new_y.index), list(new_y.columns)]
                    v_new = pd.DataFrame(v)
                    v_mask = v_new.loc[list(new_y.index), list(new_y.columns)]
                    tm_mask = tm[new_y.columns]

                    fig, ax, __ = pf.plot_adcp(tm_mask, np.array(y_mask), np.array(v_mask), ylabel, clabel, color,
                                               n_stdev=None)

                    if bin_nan > 0:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + bin_title), fontsize=8)
                    else:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1), fontsize=8)

                    sfile = '-'.join((filename, var, t0[:10]))
                    pf.save_fig(save_dir, sfile)

                    fig, ax, n_nans_all = pf.plot_adcp(tm_mask, np.array(y_mask), np.array(v_mask), ylabel, clabel, color, n_stdev=5)
                    title_i = 'removed: {} nans, {} fill values, {} extreme values, {} bad beams, {} GR [{}, {}]'.format(
                        n_nan, n_fv, n_ev, n_bb, n_grange, g_min, g_max)

                    if bin_nan > 0:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title_i + '\n' + bin_title), fontsize=8)
                    else:
                        ax.set_title((title_text + '\n' + t0 + ' - ' + t1 + '\n' + title_i), fontsize=8)

                    sfile = '-'.join((filename, var, t0[:10])) + '_rmoutliers'
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list =['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180207-CE06ISSM-MFD35-04-ADCPTM000-telemetered-adcp_velocity_earth/catalog.html',
               'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180205-CE06ISSM-MFD35-04-ADCPTM000-recovered_inst-adcp_velocity_earth/catalog.html',
               'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180204-CE06ISSM-MFD35-04-ADCPTM000-recovered_host-adcp_velocity_earth/catalog.html']

    # url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180120-CE01ISSM-MFD35-04-ADCPTM000-telemetered-adcp_velocity_earth/catalog.html',
    #             'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180115-CE01ISSM-MFD35-04-ADCPTM000-recovered_inst-adcp_velocity_earth/catalog.html',
    #             'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T180110-CE01ISSM-MFD35-04-ADCPTM000-recovered_host-adcp_velocity_earth/catalog.html']
    # url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190524T175913-CP01CNSM-MFD35-01-ADCPTF000-recovered_inst-adcp_velocity_earth/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
