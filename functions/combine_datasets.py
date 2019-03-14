#! /usr/bin/env python
import xarray as xr
import numpy as np
import pandas as pd
import functions.common as cf


def append_science_data(preferred_stream_df, n_streams, refdes, dataset_list, sci_vars_dict, et=[], stime=None, etime=None):
    # build dictionary of science data from the preferred dataset for each deployment
    for index, row in preferred_stream_df.iterrows():
        for ii in range(n_streams):
            rms = '-'.join((refdes, row[ii]))
            drms = '_'.join((row['deployment'], rms))
            print(drms)

            for d in dataset_list:
                ds_drms = d.split('/')[-1].split('_20')[0]
                if ds_drms == drms:
                    ds = xr.open_dataset(d, mask_and_scale=False)
                    ds = ds.swap_dims({'obs': 'time'})
                    if stime is not None and etime is not None:
                        ds = ds.sel(time=slice(stime, etime))
                        if len(ds['time'].values) == 0:
                            print('No data for specified time range: ({} to {})'.format(stime, etime))
                            continue

                    fmethod_stream = '-'.join((ds.collection_method, ds.stream))

                    for strm, b in sci_vars_dict.items():
                        # if the reference designator has 1 science data stream
                        if strm == 'common_stream_placeholder':
                            sci_vars_dict = append_variable_data(ds, sci_vars_dict,
                                                                 'common_stream_placeholder', et)
                        # if the reference designator has multiple science data streams
                        elif fmethod_stream in sci_vars_dict[strm]['ms']:
                            sci_vars_dict = append_variable_data(ds, sci_vars_dict, strm, et)
    return sci_vars_dict


def append_variable_data(ds, variable_dict, common_stream_name, exclude_times):
    ds_vars = cf.return_raw_vars(list(ds.data_vars.keys()) + list(ds.coords))
    vars_dict = variable_dict[common_stream_name]['vars']
    for var in ds_vars:
        try:
            long_name = ds[var].long_name
            if long_name in list(vars_dict.keys()):
                if ds[var].units == vars_dict[long_name]['db_units']:
                    if ds[var]._FillValue not in vars_dict[long_name]['fv']:
                        vars_dict[long_name]['fv'].append(ds[var]._FillValue)
                    if ds[var].units not in vars_dict[long_name]['units']:
                        vars_dict[long_name]['units'].append(ds[var].units)
                    tD = ds['time'].values
                    varD = ds[var].values
                    deployD = ds['deployment'].values
                    if len(exclude_times) > 0:
                        for et in exclude_times:
                            tD, varD, deployD = exclude_time_ranges(tD, varD, deployD, et)
                        if len(tD) > 0:
                            vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                            vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                            vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)
                    else:
                        vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                        vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                        vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)

        except AttributeError:
            continue
    return variable_dict


def common_long_names(science_variable_dictionary):
    # return dictionary of common variables
    vals_df = pd.DataFrame(science_variable_dictionary)
    vals_df_nafree = vals_df.dropna()
    vals_df_onlyna = vals_df[~vals_df.index.isin(vals_df_nafree.index)]
    if len(list(vals_df_onlyna.index)) > 0:
        print('\nWARNING: variable names that are not common among methods: {}'.format(list(vals_df_onlyna.index)))

    var_dict = dict()
    for ii, vv in vals_df_nafree.iterrows():
        units = []
        for x in range(len(vv)):
            units.append(vv[x]['db_units'])
        if len(np.unique(units)) == 1:
            var_dict.update({ii: vv[0]})

    return var_dict


def exclude_time_ranges(time_data, variable_data, deploy_data, time_lst):
    t0 = np.datetime64(time_lst[0])
    t1 = np.datetime64(time_lst[1])
    ind = np.where((time_data < t0) | (time_data > t1), True, False)
    timedata = time_data[ind]
    variabledata = variable_data[ind]
    deploydata = deploy_data[ind]
    return timedata, variabledata, deploydata


def initialize_empty_arrays(dictionary, stream_name):
    for kk, vv in dictionary[stream_name]['vars'].items():
        dictionary[stream_name]['vars'][kk].update({'t': np.array([], dtype='datetime64[ns]'),
                                                    'pressure': np.array([]),
                                                    'values': np.array([]),
                                                    'fv': [], 'units': [], 'deployments': np.array([])})
    return dictionary


def sci_var_long_names(refdes):
    # get science variable long names from the Data Review Database
    stream_sci_vars_dict = dict()
    dr = cf.refdes_datareview_json(refdes)
    for x in dr['instrument']['data_streams']:
        dr_ms = '-'.join((x['method'], x['stream_name']))
        sci_vars = dict()
        for y in x['stream']['parameters']:
            if y['data_product_type'] == 'Science Data':
                sci_vars.update({y['display_name']: dict(db_units=y['unit'], var_name=y['name'])})
        if len(sci_vars) > 0:
            stream_sci_vars_dict.update({dr_ms: sci_vars})
    return stream_sci_vars_dict


def sci_var_long_names_check(stream_sci_vars_dict):
    # check if the science variable long names are the same for each stream
    methods = []
    streams = []
    for k in list(stream_sci_vars_dict.keys()):
        methods.append(k.split('-')[0])
        streams.append(k.split('-')[1])

    # if the reference designator has one science data stream
    if (len(np.unique(methods)) > len(np.unique(streams))) or ('ctdbp' in streams[0]):
        var_dict = common_long_names(stream_sci_vars_dict)
        sci_vars_dict = dict(common_stream_placeholder=dict(vars=var_dict,
                                                            ms=list(stream_sci_vars_dict.keys())))
        sci_vars_dict = initialize_empty_arrays(sci_vars_dict, 'common_stream_placeholder')

    # if the reference designator has multiple science data streams
    else:
        method_stream_df = cf.stream_word_check(stream_sci_vars_dict)
        method_stream_df['method_stream'] = method_stream_df['method'] + '-' + method_stream_df['stream_name']
        common_stream_names = np.unique(method_stream_df['stream_name_compare'].tolist()).tolist()
        sci_vars_dict = dict()
        for csn in common_stream_names:
            check = dict()
            df = method_stream_df.loc[method_stream_df['stream_name_compare'] == csn]
            ss = df['method_stream'].tolist()
            for k, v in stream_sci_vars_dict.items():
                if k in ss:
                    check.update({k: v})

            var_dict = common_long_names(check)
            sci_vars_dict.update({csn: dict(vars=var_dict, ms=ss)})
            sci_vars_dict = initialize_empty_arrays(sci_vars_dict, csn)

    return sci_vars_dict
