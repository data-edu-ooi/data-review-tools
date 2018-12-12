#! /usr/bin/env python
import itertools
import xarray as xr
import numpy as np
import functions.common as cf


def append_science_data(preferred_stream_df, n_streams, refdes, dataset_list, sci_vars_dict, et=[]):
    # build dictionary of science data from the preferred dataset for each deployment
    for index, row in preferred_stream_df.iterrows():
        for ii in range(n_streams):
            rms = '-'.join((refdes, row[ii]))
            print('{} {}'.format(row['deployment'], rms))

            for d in dataset_list:
                ds = xr.open_dataset(d, mask_and_scale=False)
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
    ds_vars = cf.return_raw_vars(list(ds.data_vars.keys()))
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


def sci_var_long_names(refdes):
    # get science variable long names from the Data Review Database
    stream_sci_vars_dict = dict()
    dr = cf.refdes_datareview_json(refdes)
    for x in dr['instrument']['data_streams']:
        dr_ms = '-'.join((x['method'], x['stream_name']))
        sci_vars = dict()
        for y in x['stream']['parameters']:
            if y['data_product_type'] == 'Science Data':
                sci_vars.update({y['display_name']: dict(db_units=y['unit'])})
        if len(sci_vars) > 0:
            stream_sci_vars_dict.update({dr_ms: sci_vars})
    return stream_sci_vars_dict


def sci_var_long_names_check(stream_sci_vars_dict):
    # check if the science variable long names are the same for each stream
    groups = itertools.groupby(stream_sci_vars_dict.values())
    next(groups, None)
    if next(groups, None) is None:  # the reference designator has one science data stream
        sci_vars_dict = dict(common_stream_placeholder=dict(vars=list(stream_sci_vars_dict.values())[0],
                                                            ms=list(stream_sci_vars_dict.keys())))
        sci_vars_dict = initialize_empty_arrays(sci_vars_dict, 'common_stream_placeholder')
    else:  # the reference designator has multiple science data streams
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

            groups = itertools.groupby(check.values())
            next(groups, None)
            if next(groups, None) is None:
                sci_vars_dict.update({csn: dict(vars=list(check.values())[0], ms=ss)})
                sci_vars_dict = initialize_empty_arrays(sci_vars_dict, csn)
            else:
                print('Streams with common name: <{}> do not have common science variables'.format(csn))
    return sci_vars_dict
