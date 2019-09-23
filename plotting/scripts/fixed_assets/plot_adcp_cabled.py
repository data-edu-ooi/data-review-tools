#!/usr/bin/env python

"""
Created on Sept 20 2019

@author: Lori Garzio
@brief: This script is used create plots of science variables for cabled ADCPs by deployment, broken up into 4 equal
segments. Alternatively, the user has the option of selecting a specific time range and/or deployment to plot.
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import itertools
import functions.common as cf
import functions.plotting as pf


def date_range(start, end, intv):
    diff = (end - start) / intv
    for i in range(intv):
        yield(start + diff * i).strftime('%Y-%m-%d')
    yield end.strftime('%Y-%m-%d')


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
        for aa, bb in y_bad_beams.items():
            y[bb['values'] < 75] = np.nan
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


def main(sDir, url_list, start_time, end_time, deployment_num):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        deployments = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
                for ud in udatasets:
                    if ud.split('/')[-1].split('_')[0] not in deployments:
                        deployments.append(ud.split('/')[-1].split('_')[0])
        datasets = list(itertools.chain(*datasets))
        deployments.sort()

        fdatasets = np.unique(datasets).tolist()
        for deploy in deployments:
            if deployment_num is not None:
                if int(deploy[-4:]) is not deployment_num:
                    print('\nskipping {}'.format(deploy))
                    continue

            rdatasets = [s for s in fdatasets if deploy in s]

            # break deployment into 4 segments or make a list of the time range specified
            if start_time is not None and end_time is not None:
                dt_range = [dt.datetime.strftime(start_time, '%Y-%m-%d'), dt.datetime.strftime(end_time, '%Y-%m-%d')]
            else:
                # Get deployment info from the data review database
                dr_data = cf.refdes_datareview_json(r)
                d_info = [x for x in dr_data['instrument']['deployments'] if x['deployment_number'] == int(deploy[-4:])]
                d_info = d_info[0]
                deploy_start = dt.datetime.strptime(str(d_info['start_date']).split('T')[0], '%Y-%m-%d')
                deploy_stop = dt.datetime.strptime(str(d_info['stop_date']).split('T')[0], '%Y-%m-%d') + dt.timedelta(
                    days=1)
                dt_range = list(date_range(deploy_start, deploy_stop, 4))

            sci_vars_dict = {'time': dict(values=np.array([], dtype=np.datetime64), fv=[], ln=[]),
                             'bin_depths': dict(values=np.array([]), units=[], fv=[], ln=[])}
            percentgood = {'percent_good_beam1': dict(values=np.array([])),
                           'percent_good_beam2': dict(values=np.array([])),
                           'percent_good_beam3': dict(values=np.array([])),
                           'percent_good_beam4': dict(values=np.array([]))}

            for dtri in range(len(dt_range) - 1):
                stime = dt.datetime.strptime(dt_range[dtri], '%Y-%m-%d')
                etime = dt.datetime.strptime(dt_range[dtri + 1], '%Y-%m-%d')
                if len(rdatasets) > 0:
                    for i in range(len(rdatasets)):
                    #for i in range(0, 2):  ##### for testing
                        ds = xr.open_dataset(rdatasets[i], mask_and_scale=False)
                        ds = ds.swap_dims({'obs': 'time'})
                        print('\nAppending data from {}: file {} of {}'.format(deploy, i + 1, len(rdatasets)))

                        ds = ds.sel(time=slice(stime, etime))
                        if len(ds['time'].values) == 0:
                            print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                            continue

                        try:
                            print(fname)
                        except NameError:
                            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(rdatasets[0])
                            sci_vars = cf.return_science_vars(stream)
                            # drop the following list of key words from science variables list
                            sci_vars = notin_list(sci_vars, ['salinity', 'temperature', 'bin_depths', 'beam'])
                            sci_vars = [name for name in sci_vars if ds[name].units != 'mm s-1']

                            for sci_var in sci_vars:
                                sci_vars_dict.update({sci_var: dict(values=np.array([]), units=[], fv=[], ln=[])})

                        # append data for the deployment into a dictionary
                        for s_v, info in sci_vars_dict.items():
                            print(s_v)
                            vv = ds[s_v]
                            try:
                                if vv.units not in info['units']:
                                    info['units'].append(vv.units)
                            except AttributeError:
                                print('no units')
                            try:
                                if vv._FillValue not in info['fv']:
                                    info['fv'].append(vv._FillValue)
                            except AttributeError:
                                print('no fill value')

                            try:
                                if vv.long_name not in info['ln']:
                                    info['ln'].append(vv.long_name)
                            except AttributeError:
                                print('no long name')

                            if len(vv.dims) == 1:
                                info['values'] = np.append(info['values'], vv.values)
                            else:
                                if len(info['values']) == 0:
                                    info['values'] = vv.values.T
                                else:
                                    info['values'] = np.concatenate((info['values'], vv.values.T), axis=1)

                        # append percent good beams
                        for j, k in percentgood.items():
                            pgvv = ds[j]
                            fv_pgvv = pgvv._FillValue
                            pgvv = pgvv.values.T.astype(float)
                            pgvv[pgvv == fv_pgvv] = np.nan
                            if len(k['values']) == 0:
                                k['values'] = pgvv
                            else:
                                k['values'] = np.concatenate((k['values'], pgvv), axis=1)

                    array = subsite[0:2]
                    filename = '_'.join(fname.split('_')[:-1])
                    save_dir = os.path.join(sDir, array, subsite, refdes, 'plots', deployment)
                    cf.create_dir(save_dir)

                    tm = sci_vars_dict['time']['values']
                    t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
                    title_text = ' '.join((deployment, refdes, method))

                    bd = sci_vars_dict['bin_depths']
                    ylabel = 'bin_depths ({})'.format(bd['units'][0])

                    print('\nPlotting')
                    for var in sci_vars:
                        print('----{}'.format(var))
                        v = sci_vars_dict[var]
                        fv = v['fv'][0]
                        v_name = v['ln'][0]
                        units = v['units'][0]

                        if len(np.shape(v['values'])) == 1:
                            v, n_nan, n_fv, n_ev, n_grange, g_min, g_max, n_std = reject_err_data_1_dims(v['values'], fv, r, var, n=5)

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
                            v, n_nan, n_fv, n_ev, n_bb, n_grange, g_min, g_max = reject_err_data_2_dims(v['values'], percentgood, fv, r, var)

                            clabel = '{} ({})'.format(var, units)

                            # check bin depths for extreme values
                            y = bd['values']
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
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list =['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190904T191252007Z-CE02SHBP-LJ01D-05-ADCPTB104-streamed-adcp_velocity_beam/catalog.html']
    start_time = None  # dt.datetime(2014, 10, 1, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2014, 10, 10, 0, 0, 0)  # optional, set to None if plotting all data
    deployment_num = 1  # None or int
    main(sDir, url_list, start_time, end_time, deployment_num)
