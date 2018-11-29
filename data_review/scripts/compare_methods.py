#!/usr/bin/env python
"""
Created on 10/15/2018
@author Lori Garzio
@author Leila Belabbassi
@brief Compares science data from multiple delivery methods for one instrument.

Summary output:
ds0: index=0 dataset in comparison list
ds1 = index=1 dataset in comparison list

name: variable name
units: variable units
n: number of data points for the variable being analyzed
n_nan: number of NaNs in the dataset for the variable being analyzed
missing: summary of data missing from the indexed dataset

missing_data_gaps: list of time ranges where data are missing from one dataset (data are available in the other dataset)
n_missing: list of number of data points missing from each corresponding time range in missing_data_gaps
n_missing_total: total number of missing data points (data are available in the other dataset)

unit_test: pass/fail check if the units for the variables in the two datasets are the same
n_comparison: number of data points compared between the two methods
min_abs_diff: minimum absolute difference calculated between the two datasets
max_abs_diff: maximum absolute difference calculated between the two datasets
n_diff_greater_zero: count of absolute differences that are >0
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import json
from datetime import timedelta
import functions.common as cf


def compare_data(df):
    names = df.columns
    summary = dict(deployments=dict())
    for d, row in df.iterrows():
        for i, n in enumerate(names):
            ii = i + 1
            if ii > 1:
                f1 = row[n]
                try:
                    if np.isnan(f1) is True:
                        continue
                except TypeError:
                    for x in range(ii - 1):
                        f0 = row[names[x]]
                        try:
                            if np.isnan(f0) is True:
                                continue
                        except TypeError:
                            if d not in summary['deployments'].keys():
                                summary['deployments'][d] = dict(comparison=dict())
                            compare = '{} {}'.format(names[x], n)
                            if compare not in summary['deployments'][d]['comparison'].keys():
                                summary['deployments'][d]['comparison'][compare] = dict(vars=dict())

                            if len(f0) == 1:
                                ds0 = xr.open_dataset(f0[0])
                                ds0 = ds0.swap_dims({'obs': 'time'})
                            else:
                                ds0 = xr.open_mfdataset(f0)
                                ds0 = ds0.swap_dims({'obs': 'time'})
                                ds0 = ds0.chunk({'time': 100})
                            splt0 = compare.split(' ')[0].split('-')
                            ds0_sci_vars = cf.return_science_vars(splt0[1])
                            ds0_method = splt0[0]

                            if len(f1) == 1:
                                ds1 = xr.open_dataset(f1[0])
                                ds1 = ds1.swap_dims({'obs': 'time'})
                            else:
                                ds1 = xr.open_mfdataset(f1)
                                ds1 = ds1.swap_dims({'obs': 'time'})
                                ds1 = ds1.chunk({'time': 100})
                            splt1 = compare.split(' ')[1].split('-')
                            ds1_sci_vars = cf.return_science_vars(splt1[1])
                            ds1_method = splt1[0]

                            # find where the variable long names are the same
                            ds0names = long_names(ds0, ds0_sci_vars)
                            ds0names.rename(columns={'name': 'name_ds0'}, inplace=True)
                            ds1names = long_names(ds1, ds1_sci_vars)
                            ds1names.rename(columns={'name': 'name_ds1'}, inplace=True)
                            mapping = pd.merge(ds0names, ds1names, on='long_name', how='inner')
                            print('----------------------')
                            print('{}: {}'.format(d, compare))
                            print('----------------------')

                            blank_dict = {'missing_data_gaps': [], 'n_missing': [], 'n_missing_days_total': 0,
                                          'n_missing_total': 0}

                            for rr in mapping.itertuples():
                                index, name_ds0, long_name, name_ds1 = rr
                                print(long_name)

                                # Compare data from two data streams (round timestamps to the nearest second).
                                ds0_rename = '_'.join((str(name_ds0), 'ds0'))
                                [ds0_df, ds0_units, n0, n0_nan] = get_ds_variable_info(ds0, name_ds0, ds0_rename)

                                ds1_rename = '_'.join((str(name_ds1), 'ds1'))
                                [ds1_df, ds1_units, n1, n1_nan] = get_ds_variable_info(ds1, name_ds1, ds1_rename)

                                # Compare units
                                if ds0_units == ds1_units:
                                    unit_test = 'pass'
                                else:
                                    unit_test = 'fail'

                                # Merge dataframes from both methods
                                merged = pd.merge(ds0_df, ds1_df, on='time', how='outer')

                                # Drop rows where both variables are NaNs, and make sure the timestamps are in order
                                merged.dropna(subset=[ds0_rename, ds1_rename], how='all', inplace=True)
                                if len(merged) == 0:
                                    print('No valid data to compare')
                                    n_comparison = 0
                                    n_diff_g_zero = None
                                    min_diff = None
                                    max_diff = None
                                    ds0_missing_dict = 'No valid data to compare'
                                    ds1_missing_dict = 'No valid data to compare'
                                else:
                                    merged = merged.sort_values('time').reset_index(drop=True)
                                    m_intersect = merged[merged[ds0_rename].notnull() & merged[ds1_rename].notnull()]

                                    # If the number of data points for comparison is less than 1% of the smaller sample size
                                    # compare the timestamps by rounding to the nearest hour
                                    if len(m_intersect) == 0 or float(len(m_intersect))/float(min(n0, n1))*100 < 1.00:
                                        n_comparison = 0
                                        n_diff_g_zero = None
                                        min_diff = None
                                        max_diff = None

                                        utime_df0 = unique_timestamps_hour(ds0)
                                        utime_df0['ds0'] = 'ds0'
                                        utime_df1 = unique_timestamps_hour(ds1)
                                        utime_df1['ds1'] = 'ds1'
                                        umerged = pd.merge(utime_df0, utime_df1, on='time', how='outer')
                                        umerged = umerged.sort_values('time').reset_index(drop=True)

                                        if 'telemetered' in ds0_method:
                                            ds0_missing_dict = 'method not checked for missing data'
                                        else:
                                            ds0_missing = umerged.loc[umerged['ds0'].isnull()]
                                            if len(ds0_missing) > 0:
                                                ds0_missing_dict = missing_data_times(ds0_missing)
                                                if ds0_missing_dict != blank_dict:
                                                    ds0_missing_dict['n_hours_missing'] = ds0_missing_dict.pop('n_missing')
                                                    ds0_missing_dict['n_hours_missing_total'] = ds0_missing_dict.pop('n_missing_total')
                                                else:
                                                    ds0_missing_dict = 'timestamps rounded to the hour: no missing data'
                                            else:
                                                ds0_missing_dict = 'timestamps rounded to the hour: no missing data'

                                        if 'telemetered' in ds1_method:
                                            ds1_missing_dict = 'method not checked for missing data'
                                        else:
                                            ds1_missing = umerged.loc[umerged['ds1'].isnull()]
                                            if len(ds1_missing) > 0:
                                                ds1_missing_dict = missing_data_times(ds1_missing)
                                                if ds1_missing_dict != blank_dict:
                                                    ds1_missing_dict['n_hours_missing'] = ds1_missing_dict.pop('n_missing')
                                                    ds1_missing_dict['n_hours_missing_total'] = ds1_missing_dict.pop('n_missing_total')
                                                else:
                                                    ds1_missing_dict = 'timestamps rounded to the hour: no missing data'
                                            else:
                                                ds1_missing_dict = 'timestamps rounded to the hour: no missing data'

                                    else:
                                        # Find where data are available in one dataset and missing in the other if
                                        # timestamps match exactly. Don't check for missing data in telemetered
                                        # datasets.
                                        if 'telemetered' in ds0_method:
                                            ds0_missing_dict = 'method not checked for missing data'
                                        else:
                                            ds0_missing = merged.loc[merged[ds0_rename].isnull()]
                                            if len(ds0_missing) > 0:
                                                ds0_missing_dict = missing_data_times(ds0_missing)
                                                if ds0_missing_dict == blank_dict:
                                                    ds0_missing_dict = 'no missing data'
                                            else:
                                                ds0_missing_dict = 'no missing data'

                                        if 'telemetered' in ds1_method:
                                            ds1_missing_dict = 'method not checked for missing data'
                                        else:
                                            ds1_missing = merged.loc[merged[ds1_rename].isnull()]
                                            if len(ds1_missing) > 0:
                                                ds1_missing_dict = missing_data_times(ds1_missing)
                                                if ds1_missing_dict == blank_dict:
                                                    ds1_missing_dict = 'no missing data'
                                            else:
                                                ds1_missing_dict = 'no missing data'

                                        # Where the data intersect, calculate the difference between the methods
                                        diff = m_intersect[ds0_rename] - m_intersect[ds1_rename]
                                        n_diff_g_zero = sum(abs(diff) > 0.99999999999999999)

                                        min_diff = round(min(abs(diff)), 10)
                                        max_diff = round(max(abs(diff)), 10)
                                        n_comparison = len(diff)

                                summary['deployments'][d]['comparison'][compare]['vars'][str(long_name)] = dict(
                                    ds0=dict(name=name_ds0, units=ds0_units, n=n0, n_nan=n0_nan, missing=ds0_missing_dict),
                                    ds1=dict(name=name_ds1, units=ds1_units, n=n1, n_nan=n1_nan, missing=ds1_missing_dict),
                                    unit_test=unit_test, n_comparison=n_comparison, n_diff_greater_zero=n_diff_g_zero,
                                    min_abs_diff=min_diff, max_abs_diff=max_diff)
    return summary


def get_ds_variable_info(dataset, variable_name, rename):
    ds_df = pd.DataFrame({'time': dataset['time'].data, variable_name: dataset[variable_name].data})
    ds_units = var_units(dataset[variable_name])
    ds_df.rename(columns={str(variable_name): rename}, inplace=True)
    n = len(ds_df[rename])
    n_nan = sum(ds_df[rename].isnull())

    # round to the nearest second
    ds_df['time'] = ds_df['time'].map(lambda t: t.replace(microsecond=0) + timedelta(seconds=(round(t.microsecond / 1000000.0))))

    return [ds_df, ds_units, n, n_nan]


def long_names(dataset, vars):
    name = []
    long_name = []
    for v in vars:
        name.append(v)  # list of recovered variable names

        try:
            longname = dataset[v].long_name
        except AttributeError:
            longname = vars

        long_name.append(longname)

    return pd.DataFrame({'name': name, 'long_name': long_name})


def merge_two_dicts(dict1, dict2):
    # add info from dict2 to dict1
    if len(dict1) == 0:
        dict1.update(dict2)
    else:
        for d in dict2['deployments'].keys():
            if d not in dict1['deployments'].keys():
                dict1['deployments'][d] = dict2['deployments'][d]
            else:
                dict1['deployments'][d]['comparison'].update(dict2['deployments'][d]['comparison'])
    return dict1


def missing_data_times(df):
    # return a dictionary of time ranges, number of data points and number of days where data are missing (but available
    # in a comparable dataset). skips gaps that are only 1 data point (or one hour if data are rounded to the hour).
    md_list = []
    n_list = []
    mdays = []
    index_break = []
    ilist = df.index.tolist()

    if len(ilist) == 1:
        ii = ilist[0]
        md_list.append(pd.to_datetime(str(df['time'][ii])).strftime('%Y-%m-%dT%H:%M:%S'))
        n_list.append(1)
    else:
        for i, n in enumerate(ilist):
            if i == 0:
                index_break.append(ilist[i])
            elif i == (len(ilist) - 1):
                index_break.append(ilist[i])
            else:
                if (n - ilist[i-1]) > 1:
                    index_break.append(ilist[i-1])
                    index_break.append(ilist[i])

        for ii, nn in enumerate(index_break):
            if ii % 2 == 0:  # check that the index is an even number
                if index_break[ii + 1] != nn:  # only list gaps that are more than 1 data point
                    try:
                        # create a list of timestamps for each gap to get the unique # of days missing from one dataset
                        time_lst = [df['time'][t].date() for t in range(nn, index_break[ii + 1] + 1)]
                    except KeyError:  # if the last data gap is only 1 point, skip
                        continue
                    md_list.append([pd.to_datetime(str(df['time'][nn])).strftime('%Y-%m-%dT%H:%M:%S'),
                                    pd.to_datetime(str(df['time'][index_break[ii + 1]])).strftime('%Y-%m-%dT%H:%M:%S')])
                    n_list.append(index_break[ii + 1] - nn + 1)
                    mdays.append(len(np.unique(time_lst)))

    n_total = sum(n_list)
    n_days = sum(mdays)

    return dict(missing_data_gaps=md_list, n_missing=n_list, n_missing_total=n_total, n_missing_days_total=n_days)


def unique_timestamps_hour(ds):
    # return dataframe of the unique timestamps rounded to the nearest hour
    df = pd.DataFrame(ds['time'].data, columns=['time'])
    df = df['time'].map(lambda t: t.replace(second=0, microsecond=0, nanosecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30))
    udf = pd.DataFrame(np.unique(df), columns=['time'])

    return udf


def var_units(variable):
    try:
        y_units = variable.units
    except AttributeError:
        y_units = 'no_units'

    return y_units


def main(sDir, url_list):
    # get summary lists of reference designators and delivery methods
    rd_list = []
    rdm_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        rdm = '-'.join((rd, elements[5]))
        if rd not in rd_list:
            rd_list.append(rd)
        if rdm not in rdm_list:
            rdm_list.append(rdm)

    json_file_list = []
    for r in rd_list:
        rdm_filtered = [k for k in rdm_list if r in k]
        dinfo = {}
        save_dir = os.path.join(sDir, r.split('-')[0], r)
        cf.create_dir(save_dir)
        sfile = os.path.join(save_dir, '{}-method_comparison.json'.format(r))
        if len(rdm_filtered) == 1:
            print('Only one delivery method provided - no comparison.')
            dinfo['note'] = 'no comparison - only one delivery method provided'
            with open(sfile, 'w') as outfile:
                json.dump(dinfo, outfile)
            json_file_list.append(str(sfile))
            continue

        elif len(rdm_filtered) > 1 & len(rdm_filtered) <= 3:
            print('\nComparing data from different methods for: {}'.format(r))
            for i in range(len(rdm_filtered)):
                urls = [x for x in url_list if rdm_filtered[i] in x]
                for u in urls:
                    splitter = u.split('/')[-2].split('-')
                    catalog_rms = '-'.join((r, splitter[-2], splitter[-1]))
                    udatasets = cf.get_nc_urls([u])
                    idatasets = []
                    for dss in udatasets:  # filter out collocated data files
                        if catalog_rms in dss.split('/')[-1].split('_20')[0]:
                            idatasets.append(dss)
                    deployments = [str(k.split('/')[-1][0:14]) for k in idatasets]
                    udeploy = np.unique(deployments).tolist()
                    for ud in udeploy:
                        rdatasets = [s for s in idatasets if ud in s]
                        file_ms_lst = []
                        for dataset in rdatasets:
                            splt = dataset.split('/')[-1].split('_20')[0].split('-')
                            file_ms_lst.append('-'.join((splt[-2], splt[-1])))
                        file_ms = np.unique(file_ms_lst).tolist()[0]
                        try:
                            dinfo[file_ms]
                        except KeyError:
                            dinfo[file_ms] = {}
                        dinfo[file_ms].update({ud: rdatasets})

        else:
            print('More than 3 methods provided. Please provide fewer datasets for analysis.')
            continue

        dinfo_df = pd.DataFrame(dinfo)

        umethods = []
        ustreams = []
        for k in dinfo.keys():
            umethods.append(k.split('-')[0])
            ustreams.append(k.split('-')[1])

        if len(np.unique(ustreams)) > len(np.unique(umethods)):  # if there is more than 1 stream per delivery method
            mdict = dict()
            method_stream_df = cf.stream_word_check(dinfo)
            for cs in (np.unique(method_stream_df['stream_name_compare'])).tolist():
                print('Common stream_name: {}'.format(cs))
                method_stream_list = []
                for row in method_stream_df.itertuples():
                    index, method, stream_name, stream_name_compare = row
                    if stream_name_compare == cs:
                        method_stream_list.append('-'.join((method, stream_name)))
                dinfo_df_filtered = dinfo_df[method_stream_list]
                summary_dict = compare_data(dinfo_df_filtered)

                # merge dictionaries for all streams for one reference designator
                mdict = merge_two_dicts(mdict, summary_dict)
                with open(sfile, 'w') as outfile:
                    json.dump(mdict, outfile)

        else:
            summary_dict = compare_data(dinfo_df)
            with open(sfile, 'w') as outfile:
                json.dump(summary_dict, outfile)

        json_file_list.append(str(sfile))

    return json_file_list


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150658-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150707-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150716-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150726-GP03FLMA-RIM01-02-CTDMOG041-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html']

    main(sDir, url_list)
