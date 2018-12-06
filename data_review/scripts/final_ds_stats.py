#!/usr/bin/env python
"""
Created on 11/12/2018
@author Lori Garzio
@brief Calculate final statistics for science variables in a dataset. Long Names and units for science variables must
be the same among methods.
sDir: location to save summary output
url: list of THREDDs urls containing .nc files to analyze.
"""

import os
from urllib.request import urlopen
import itertools
import xarray as xr
import pandas as pd
import re
import numpy as np
import json
import ast
import functions.common as cf


def append_variable_data(ds, variable_dict, common_stream_name, exclude_times):
    ds_vars = eliminate_common_variables(list(ds.data_vars.keys()))
    vars_dict = variable_dict[common_stream_name]['vars']
    for var in ds_vars:
        try:
            long_name = ds[var].long_name
            if long_name in list(vars_dict.keys()):
                if ds[var].units == vars_dict[long_name]['db_units']:
                    vars_dict[long_name]['fv'].append(ds[var]._FillValue)
                    vars_dict[long_name]['units'].append(ds[var].units)
                    tD = ds['time'].data
                    varD = ds[var].data
                    if len(exclude_times) > 0:
                        for et in exclude_times:
                            tD, varD = exclude_time_ranges(tD, varD, et)
                        vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                        vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                    else:
                        vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                        vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)

        except AttributeError:
            continue

    return variable_dict


def eliminate_common_variables(vlist):
    common = ['quality_flag', 'provenance', 'deployment', 'timestamp', 'qc_executed', 'qc_results']
    regex = re.compile('|'.join(common))
    varlist = [s for s in vlist if not regex.search(s)]
    return varlist


def exclude_time_ranges(time_data, variable_data, time_lst):
    t0 = np.datetime64(time_lst[0])
    t1 = np.datetime64(time_lst[1])
    ind = np.where((time_data < t0) | (time_data > t1), True, False)
    timedata = time_data[ind]
    variabledata = variable_data[ind]
    return timedata, variabledata


def initialize_empty_arrays(dictionary, stream_name):
    for kk, vv in dictionary[stream_name]['vars'].items():
        dictionary[stream_name]['vars'][kk].update({'t': np.array([], dtype='datetime64[ns]'), 'values': np.array([]),
                                                    'fv': [], 'units': []})
    return dictionary


def main(sDir, url_list):
    eliminate_times = pd.read_csv('/Users/lgarzio/Documents/OOI/DataReviews/test4/eliminate_times.csv')
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        save_dir = os.path.join(sDir, r.split('-')[0], r)
        cf.create_dir(save_dir)

        et = ast.literal_eval(eliminate_times.loc[eliminate_times['refdes'] == r]['times'][0])

        # get science variable long names from the Data Review Database
        stream_sci_vars = dict()
        dr = cf.refdes_datareview_json(r)
        for x in dr['instrument']['data_streams']:
            dr_ms = '-'.join((x['method'], x['stream_name']))
            sci_vars = dict()
            for y in x['stream']['parameters']:
                if y['data_product_type'] == 'Science Data':
                    sci_vars.update({y['display_name']: dict(db_units=y['unit'])})
            if len(sci_vars) > 0:
                stream_sci_vars.update({dr_ms: sci_vars})

        # check if the science variable long names are the same for each stream
        groups = itertools.groupby(stream_sci_vars.values())
        next(groups, None)
        if next(groups, None) is None:  # the reference designator has one science data stream
            sci_vars_dict = dict(common_stream_placeholder=dict(vars=list(stream_sci_vars.values())[0],
                                                                ms=list(stream_sci_vars.keys())))
            sci_vars_dict = initialize_empty_arrays(sci_vars_dict, 'common_stream_placeholder')
        else:  # the reference designator has multiple science data streams
            method_stream_df = cf.stream_word_check(stream_sci_vars)
            method_stream_df['method_stream'] = method_stream_df['method'] + '-' + method_stream_df['stream_name']
            common_stream_names = np.unique(method_stream_df['stream_name_compare'].tolist()).tolist()
            sci_vars_dict = dict()
            for csn in common_stream_names:
                check = dict()
                df = method_stream_df.loc[method_stream_df['stream_name_compare'] == csn]
                ss = df['method_stream'].tolist()
                for k, v in stream_sci_vars.items():
                    if k in ss:
                        check.update({k: v})

                groups = itertools.groupby(check.values())
                next(groups, None)
                if next(groups, None) is None:
                    sci_vars_dict.update({csn: dict(vars=list(check.values())[0], ms=ss)})
                    sci_vars_dict = initialize_empty_arrays(sci_vars_dict, csn)
                else:
                    print('Streams with common name: <{}> do not have common science variables'.format(csn))

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        for index, row in ps_df.iterrows():
            for ii in range(n_streams):
                rms = '-'.join((r, row[ii]))
                print('{} {}'.format(row['deployment'], rms))

                for u in url_list:
                    splitter = u.split('/')[-2].split('-')
                    rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
                    catalog_rms = '-'.join((rd_check, splitter[-2], splitter[-1]))

                    # complete the analysis by reference designator
                    if catalog_rms == rms:
                        udatasets = cf.get_nc_urls([u])
                        rdatasets = [s for s in udatasets if row['deployment'] in s]
                        if len(rdatasets) > 0:
                            datasets = []
                            for dss in rdatasets:  # filter out collocated data files
                                if catalog_rms in dss.split('/')[-1].split('_20')[0]:
                                    datasets.append(dss)

                            for dataset in datasets:
                                ds = xr.open_dataset(dataset, mask_and_scale=False)
                                fmethod_stream = '-'.join((ds.collection_method, ds.stream))

                                for strm, b in sci_vars_dict.items():
                                    # if the reference designator has 1 science data stream
                                    if strm == 'common_stream_placeholder':
                                        sci_vars_dict = append_variable_data(ds, sci_vars_dict,
                                                                             'common_stream_placeholder', et)
                                    # if the reference designator has multiple science data streams
                                    elif fmethod_stream in sci_vars_dict[strm]['ms']:
                                        sci_vars_dict = append_variable_data(ds, sci_vars_dict, strm, et)

        # analyze combined dataset
        print('\nAnalyzing combined dataset and writing summary file')
        headers = ['common_stream_name', 'potential_methods_streams', 'long_name', 'units', 't0', 't1', 'fill_value',
                   'n_all', 'n_outliers', 'n_nans', 'n_fillvalues', 'n_stats', 'mean', 'min', 'max', 'stdev']
        rows = []
        for m, n in sci_vars_dict.items():
            print(m)
            for sv, vinfo in n['vars'].items():
                print(sv)
                lst_fill_value = np.unique(vinfo['fv']).tolist()
                if len(lst_fill_value) == 1:
                    fill_value = lst_fill_value[0]
                else:
                    print('No unique fill value for {}'.format(sv))

                lunits = np.unique(vinfo['units']).tolist()

                t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                data = vinfo['values']
                n_all = len(data)

                # reject NaNs
                data_nonan = data[~np.isnan(data)]
                n_nan = n_all - len(data_nonan)

                # reject fill values
                data_nonan_nofv = data_nonan[data_nonan != fill_value]
                n_fv = n_all - n_nan - len(data_nonan_nofv)

                if len(data_nonan_nofv) > 1:
                    [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_nonan_nofv, 3)
                else:
                    num_outliers = None
                    mean = None
                    vmin = None
                    vmax = None
                    sd = None
                    n_stats = None

                rows.append([m, n['ms'], sv, lunits, t0, t1, lst_fill_value, n_all, num_outliers, n_nan,
                             n_fv, n_stats, mean, vmin, vmax, sd])

        fsum = pd.DataFrame(rows, columns=headers)
        fsum.to_csv('{}/{}_final_stats.csv'.format(save_dir, r), index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123336-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123345-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123354-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']

    main(sDir, url_list)
