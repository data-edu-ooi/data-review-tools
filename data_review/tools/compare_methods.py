#!/usr/bin/env python
"""
Created on 10/15/2018
@author Lori Garzio
@author Leila Belabbassi
@brief Compares science data from multiple delivery methods for one instrument.

Summary output:
ds0: index=0 dataset in comparison list
var0: variable name
var0_units: variable units
n_ds0: number of data points for the variable being analyzed
n_ds0_nan: number of NaNs in the dataset for the variable being analyzed

ds1 = index=1 dataset in comparison list
var1: variable name
var1_units: variable units
n_ds1: number of data points for the variable being analyzed
n_ds1_nan: number of NaNs in the dataset for the variable being analyzed

unit_test: pass/fail check if the units for the variables in the two datasets are the same
n_comparison: number of data points compared between the two methods
min_abs_diff: minimum absolute difference calculated between the two datasets
max_abs_diff: maximum absolute difference calculated between the two datasets
n_diff_greater_zero: count of absolute differences that are >0

ds0_missing: summary of data missing from the index=0 dataset
ds1_missing: summary of data missing from the index=1 dataset

missing_data_gaps: list of time ranges where data are missing from one dataset (data are available in the other dataset)
n_missing: list of number of data points missing from each corresponding time range in missing_data_gaps
n_missing_total: total number of missing data points (data are available in the other dataset)
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import json
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

                            ds0 = xr.open_dataset(f0)
                            ds0_sci_vars = cf.return_science_vars(ds0.stream)
                            ds0_method = compare.split(' ')[0].split('-')[0]
                            ds1 = xr.open_dataset(f1)
                            ds1_sci_vars = cf.return_science_vars(ds1.stream)
                            ds1_method = compare.split(' ')[1].split('-')[0]

                            # find where the variable long names are the same
                            ds0names = long_names(ds0, ds0_sci_vars)
                            ds0names.rename(columns={'name': 'name_ds0'}, inplace=True)
                            ds1names = long_names(ds1, ds1_sci_vars)
                            ds1names.rename(columns={'name': 'name_ds1'}, inplace=True)
                            mapping = pd.merge(ds0names, ds1names, on='long_name', how='inner')
                            print '----------------------'
                            print d
                            print '----------------------'

                            for rr in mapping.itertuples():
                                index, long_name, name_ds0, name_ds1 = rr
                                print long_name

                                # Compare data from two data streams (cut timestamps to the nearest second).
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
                                m = pd.merge(ds0_df, ds1_df, on='time', how='outer')

                                # Drop rows where both variables are NaNs, and make sure the timestamps are in order
                                m.dropna(subset=[[ds0_rename, ds1_rename]], how='all', inplace=True)
                                m = m.sort_values('time').reset_index(drop=True)

                                # Find where data are available in one dataset and missing in the other
                                ds0_missing = m.loc[m[ds0_rename].isnull()]
                                if len(ds0_missing) > 0:
                                    ds0_missing_dict = missing_data_times(ds0_missing, ds0_method)
                                else:
                                    ds0_missing_dict = 'no missing data'

                                ds1_missing = m.loc[m[ds1_rename].isnull()]
                                if len(ds1_missing) > 0:
                                    ds1_missing_dict = missing_data_times(ds1_missing, ds1_method)
                                else:
                                    ds1_missing_dict = 'no missing data'

                                # Where the data intersect, calculate the difference between the methods
                                m = m[m[ds0_rename].notnull() & m[ds1_rename].notnull()]
                                diff = m[ds0_rename] - m[ds1_rename]
                                n_diff_g_zero = len(diff) - sum(abs(diff) >= 0)

                                min_diff = round(min(abs(diff)), 10)
                                max_diff = round(max(abs(diff)), 10)

                                summary['deployments'][d]['comparison'][compare]['vars'][str(long_name)] = dict(
                                    var0=name_ds0, var0_units=ds0_units, var1=name_ds1, var1_units=ds1_units,
                                    unit_test=unit_test, n_comparison=len(diff), n_diff_greater_zero=n_diff_g_zero,
                                    min_abs_diff=min_diff, max_abs_diff=max_diff, n_ds0=n0, n_ds0_nan=n0_nan,
                                    n_ds1=n1, n_ds1_nan=n1_nan, ds0_missing=ds0_missing_dict,
                                    ds1_missing=ds1_missing_dict)
    return summary


def get_ds_variable_info(dataset, variable_name, rename):
    ds_df = dataset[variable_name].to_dataframe()
    ds_units = var_units(dataset[variable_name])
    ds_df.rename(columns={str(variable_name): rename}, inplace=True)
    n = len(ds_df[rename])
    n_nan = sum(ds_df[rename].isnull())
    ds_df['time'] = ds_df['time'].map(lambda time: time.strftime('%Y-%m-%d %H:%M:%S'))

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


def missing_data_times(df, method):
    md_list = []
    n_list = []
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
                md_list.append([pd.to_datetime(str(df['time'][nn])).strftime('%Y-%m-%dT%H:%M:%S'),
                                pd.to_datetime(str(df['time'][index_break[ii + 1]])).strftime('%Y-%m-%dT%H:%M:%S')])
                n_list.append(index_break[ii + 1] - nn + 1)

    n_total = sum(n_list)

    # don't print out each data gap for telemetered data, because it's usually way too much
    if method == 'telemetered':
        md_list = '{} data gaps'.format(len(md_list))
        n_list = '{} data gaps'.format(len(n_list))

    return dict(missing_data_gaps=md_list, n_missing=n_list, n_missing_total=n_total)


def var_units(variable):
    try:
        y_units = variable.units
    except AttributeError:
        y_units = 'no_units'
    return y_units


def word_check(method_stream_dict):
    # check stream names for cases where extra words are used in the names
    omit_word = ['_dcl', '_imodem', '_conc']
    mm = []
    ss = []
    ss_new = []

    for y in method_stream_dict.keys():
        mm.append(str(y).split('-')[0])
        ss.append(str(y).split('-')[1])

    for s in ss:
        wordi = []
        for word in omit_word:
            if word in s:
                wordi.append(word)
                break

        if wordi:
            fix = s.split(wordi[0])
            if len(fix) == 2:
                ss_new.append(fix[0] + fix[1].split('_recovered')[0])
        else:
            ss_new.append(s)
    return pd.DataFrame({'method': mm, 'stream_name': ss, 'stream_name_compare': ss_new})


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
        if len(rdm_filtered) == 1:
            print 'Only one delivery method provided - no comparison.'
            continue
        elif len(rdm_filtered) > 1 & len(rdm_filtered) <= 3:
            save_dir = os.path.join(sDir, r.split('-')[0], r)
            cf.create_dir(save_dir)
            print 'Comparing data from different methods for: {}'.format(r)
            for i in range(len(rdm_filtered)):
                urls = [x for x in url_list if rdm_filtered[i] in x]
                for u in urls:
                    splitter = u.split('/')[-2].split('-')
                    catalog_rms = '-'.join((r, splitter[-2], splitter[-1]))
                    datasets = cf.get_nc_urls([u])
                    for dataset in datasets:
                        fname, subsite, refdes, method, data_stream, deployment = cf.nc_attributes(dataset)
                        file_rms = '-'.join((refdes, method, data_stream))
                        file_ms = '-'.join((method, data_stream))
                        if file_rms == catalog_rms:
                            try:
                                dinfo[file_ms]
                            except KeyError:
                                dinfo[file_ms] = {}
                            dinfo[file_ms].update({deployment: dataset})
        else:
            print 'More than 3 methods provided. Please provide fewer datasets for analysis.'
            continue

        sfile = os.path.join(save_dir, '{}-method_comparison.json'.format(r))
        dinfo_df = pd.DataFrame(dinfo)
        mdict = dict()
        if len(dinfo) > 2:  # if there is more than 1 stream per delivery method
            method_stream_df = word_check(dinfo)
            for cs in (np.unique(method_stream_df['stream_name_compare'])).tolist():
                print 'Common stream_name: {}'.format(cs)
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
