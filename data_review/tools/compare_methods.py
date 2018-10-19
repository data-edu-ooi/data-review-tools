#!/usr/bin/env python
"""
Created on 10/15/2018
@author Lori Garzio
@brief Compares science data from multiple delivery methods for one instrument.
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import json
import functions.common as cf


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


def missing_data_times(df):
    md_list = []
    n_list = []
    index_break = []
    ilist = df.index.tolist()

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

    return dict(missing_data_gaps=md_list, n_missing=n_list, n_missing_total=sum(n_list))


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
        print r
        rdm_filtered = [k for k in rdm_list if r in k]
        dinfo = {}
        if len(rdm_filtered) == 1:
            print 'Only one delivery method provided - no comparison.'
            continue
        elif len(rdm_filtered) > 1 & len(rdm_filtered) <= 3:
            save_dir = os.path.join(sDir, r.split('-')[0], r)
            cf.create_dir(save_dir)
            for i in range(len(rdm_filtered)):
                u = [x for x in url_list if rdm_filtered[i] in x][0]
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

        df = pd.DataFrame(dinfo)

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
                        for x in range(ii-1):
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
                                ds1 = xr.open_dataset(f1)
                                ds1_sci_vars = cf.return_science_vars(ds1.stream)

                                # find where the variable long names are the same
                                ds0names = long_names(ds0, ds0_sci_vars)
                                ds0names.rename(columns={'name': 'name_ds0'}, inplace=True)
                                ds1names = long_names(ds1, ds1_sci_vars)
                                ds1names.rename(columns={'name': 'name_ds1'}, inplace=True)
                                mapping = pd.merge(ds0names, ds1names, on='long_name', how='inner')

                                for rr in mapping.itertuples():
                                    index, long_name, name_ds0, name_ds1 = rr

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

                                    # Find where data are missing in one dataset and available in the other
                                    ds0_missing = m.loc[m[ds0_rename].isnull()]
                                    if len(ds0_missing) > 0:
                                        ds0_missing_dict = missing_data_times(ds0_missing)
                                    else:
                                        ds0_missing_dict = 'no missing data'

                                    ds1_missing = m.loc[m[ds1_rename].isnull()]
                                    if len(ds1_missing) > 0:
                                        ds1_missing_dict = missing_data_times(ds1_missing)
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

        sfile = os.path.join(save_dir, '{}-method_comparison.json'.format(r))
        with open(sfile, 'w') as outfile:
            json.dump(summary, outfile)

        json_file_list.append(str(sfile))

    return json_file_list


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150658-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150707-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150716-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181001T150726-GP03FLMA-RIM01-02-CTDMOG041-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html']
    main(sDir, url_list)
