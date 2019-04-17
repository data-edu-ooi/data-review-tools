#! /usr/bin/env python
import xarray as xr
import numpy as np
import pandas as pd
import functions.common as cf
import functions.plotting as pf
import os
from os import listdir
from os.path import isfile, join

def append_science_data(preferred_stream_df, n_streams, refdes, dataset_list, sci_vars_dict, et=[], stime=None, etime=None):
    # build dictionary of science data from the preferred dataset for each deployment
    for index, row in preferred_stream_df.iterrows():
        for ii in range(n_streams):
            try:
                rms = '-'.join((refdes, row[ii]))
                drms = '_'.join((row['deployment'], rms))
                print(drms)
            except TypeError:
                continue

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
                            variable_dict, pressure_unit, pressure_name = append_variable_data(ds, sci_vars_dict,
                                                                            'common_stream_placeholder', et)
                        # if the reference designator has multiple science data streams
                        elif fmethod_stream in sci_vars_dict[strm]['ms']:
                            variable_dict, pressure_unit, pressure_name = append_variable_data(ds, sci_vars_dict,
                                                                                               strm, et)


    return sci_vars_dict, pressure_unit, pressure_name


def append_variable_data(ds, variable_dict, common_stream_name, exclude_times):
    pressure_unit, pressure_name = [], []
    ds_vars = cf.return_raw_vars(list(ds.data_vars.keys()) + list(ds.coords))
    vars_dict = variable_dict[common_stream_name]['vars']
    for var in ds_vars:
        try:
            long_name = ds[var].long_name
            x = [x for x in list(vars_dict.keys()) if long_name in x]
            if len(x) != 0:
                long_name = x[0]
                if ds[var].units == vars_dict[long_name]['db_units']:
                    if ds[var]._FillValue not in vars_dict[long_name]['fv']:
                        vars_dict[long_name]['fv'].append(ds[var]._FillValue)
                    if ds[var].units not in vars_dict[long_name]['units']:
                        vars_dict[long_name]['units'].append(ds[var].units)
                    tD = ds['time'].values
                    varD = ds[var].values
                    deployD = ds['deployment'].values

                    # find the pressure to use from the data file
                    pD, p_unit, p_name = cf.add_pressure_to_dictionary_of_sci_vars(ds)
                    if p_unit not in pressure_unit:
                        pressure_unit.append(p_unit)
                    if p_name not in pressure_name:
                        pressure_name.append(p_name)

                    if len(exclude_times) > 0:
                        for et in exclude_times:
                            tD, pD, varD, deployD = exclude_time_ranges(tD, pD, varD, deployD, et)
                        if len(tD) > 0:
                            vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                            vars_dict[long_name]['pressure'] = np.append(vars_dict[long_name]['pressure'], pD)
                            vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                            vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)
                    else:
                        vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                        vars_dict[long_name]['pressure'] = np.append(vars_dict[long_name]['pressure'], pD)
                        vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                        vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)


        except AttributeError:
            continue

    return variable_dict, pressure_unit, pressure_name


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


def exclude_time_ranges(time_data, pressure_data, variable_data, deploy_data, time_lst):
    t0 = np.datetime64(time_lst[0])
    t1 = np.datetime64(time_lst[1])
    ind = np.where((time_data < t0) | (time_data > t1), True, False)
    timedata = time_data[ind]
    pressuredata = pressure_data[ind]
    variabledata = variable_data[ind]
    deploydata = deploy_data[ind]
    return timedata, pressuredata, variabledata, deploydata


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


def var_long_names(refdes):
    # get science variable long names from the Data Review Database
    stream_vars_dict = dict()
    dr = cf.refdes_datareview_json(refdes)
    for x in dr['instrument']['data_streams']:
        dr_ms = '-'.join((x['method'], x['stream_name']))
        sci_vars = dict()
        for y in x['stream']['parameters']:
            if (y['data_product_type'] == 'Science Data') or (y['data_product_type'] == 'Unprocessed Data'):
                sci_vars.update({y['display_name']: dict(db_units=y['unit'], var_name=y['name'])})
        if len(sci_vars) > 0:
            stream_vars_dict.update({dr_ms: sci_vars})
    return stream_vars_dict

def append_evaluated_science_data(sDir, preferred_stream_df, n_streams, refdes, dataset_list, sci_vars_dict, zdbar, stime=None, etime=None):
    # build dictionary of science data from the preferred dataset for each deployment
    for index, row in preferred_stream_df.iterrows():
        for ii in range(n_streams):
            try:
                rms = '-'.join((refdes, row[ii]))
                drms = '_'.join((row['deployment'], rms))
                print('\n' + drms)
            except TypeError:
                continue

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
                            variable_dict, pressure_unit, pressure_name, l0 = append_evaluated_data(
                                                                        sDir, row['deployment'], ds, sci_vars_dict,
                                                                        'common_stream_placeholder', zdbar)

                        # if the reference designator has multiple science data streams
                        elif fmethod_stream in sci_vars_dict[strm]['ms']:
                            variable_dict, pressure_unit, pressure_name, l0 = append_evaluated_data(sDir, row['deployment'],
                                                                        ds, sci_vars_dict, strm, zdbar)


    return sci_vars_dict, pressure_unit, pressure_name, l0


def append_evaluated_data(sDir, deployment, ds, variable_dict, common_stream_name, zdbar):
    pressure_unit, pressure_name = [], []
    r = '{}-{}-{}'.format(ds.subsite, ds.node, ds.sensor)
    ds_vars = cf.return_raw_vars(list(ds.data_vars.keys()) + list(ds.coords))
    vars_dict = variable_dict[common_stream_name]['vars']
    total_len = 0
    for var in ds_vars:
        try:
            long_name = ds[var].long_name
            x = [x for x in list(vars_dict.keys()) if long_name in x]
            if len(x) != 0:
                long_name = x[0]
                if ds[var].units == vars_dict[long_name]['db_units']:
                    print(var)
                    if ds[var]._FillValue not in vars_dict[long_name]['fv']:
                        vars_dict[long_name]['fv'].append(ds[var]._FillValue)
                    if ds[var].units not in vars_dict[long_name]['units']:
                        vars_dict[long_name]['units'].append(ds[var].units)
                    tD = ds['time'].values
                    varD = ds[var].values
                    deployD = ds['deployment'].values

                    # find the pressure to use from the data file
                    pD, p_unit, p_name = cf.add_pressure_to_dictionary_of_sci_vars(ds)
                    if p_unit not in pressure_unit:
                        pressure_unit.append(p_unit)
                    if p_name not in pressure_name:
                        pressure_name.append(p_name)


                    l0 = len(tD)
                    # reject erroneous data
                    tD, pD, varD, deployD = reject_erroneous_data(r, var, tD, pD, varD, deployD, ds[var]._FillValue)
                    l_erroneous = len(tD)
                    print('{} erroneous data'.format(l0 - l_erroneous))

                    if l_erroneous != 0:
                        # reject time range from data portal file export
                        tD, pD, varD, deployD = reject_timestamps_data_portal(ds.subsite, r, tD, pD, varD, deployD)
                        l_portal = len(tD)
                        print('{} suspect  - data portal'.format(l_erroneous - l_portal))
                        if l_portal != 0:

                            # reject timestamps from stat analysis
                            Dpath = '{}/{}/{}/{}/{}'.format(sDir, ds.subsite[0:2], ds.subsite, r, 'time_to_exclude')
                            tD, pD, varD, deployD = reject_timestamps_from_stat_analysis(
                                                                           Dpath, deployment, var, tD, pD,varD, deployD)
                            l_stat = len(tD)
                            print('{} suspect  - stat analysis'.format(l_portal - l_stat))

                            # # reject timestamps in a depth range
                            tD, pD, varD, deployD = reject_data_in_depth_range(tD, pD, varD, deployD, zdbar)
                            l_zrange = len(tD)
                            print('{} suspect - water depth > {} dbar'.format(l_stat - l_zrange, zdbar))

                            print(l0, ' Start < - > Final ', len(tD), 'Dictinary entry: ', len(vars_dict[long_name]['t']))
                            print()
                            vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                            vars_dict[long_name]['pressure'] = np.append(vars_dict[long_name]['pressure'], pD)
                            vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                            vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)
                        else:
                            print('suspect data - rejected all, see data portal')
                    else:
                        print('erroneous data - rejected all')
                        vars_dict[long_name]['t'] = np.append(vars_dict[long_name]['t'], tD)
                        vars_dict[long_name]['pressure'] = np.append(vars_dict[long_name]['pressure'], pD)
                        vars_dict[long_name]['values'] = np.append(vars_dict[long_name]['values'], varD)
                        vars_dict[long_name]['deployments'] = np.append(vars_dict[long_name]['deployments'], deployD)

                total_len += l0

        except AttributeError:
            continue

    return variable_dict, pressure_unit, pressure_name, total_len


def reject_erroneous_data(r, v, t, y, z, d, fz):

    """
    :param r: reference designator
    :param v: data parameter name
    :param t: time array
    :param y: pressure array
    :param z: data values
    :param d: deployment number
    :param fz: fill values defined in the data file
    :return: filtered data from fill values, NaNs, extreme values '|1e7|' and data outside global ranges
    """

    # reject fill values
    fv_ind = z != fz
    y_nofv = y[fv_ind]
    t_nofv = t[fv_ind]
    z_nofv = z[fv_ind]
    d_nofv = d[fv_ind]
    print(len(z) - len(z_nofv), ' fill values')

    # reject NaNs
    nan_ind = ~np.isnan(z_nofv)
    t_nofv_nonan = t_nofv[nan_ind]
    y_nofv_nonan = y_nofv[nan_ind]
    z_nofv_nonan = z_nofv[nan_ind]
    d_nofv_nonan = d_nofv[nan_ind]
    print(len(z_nofv) - len(z_nofv_nonan), ' NaNs')

    # reject extreme values
    ev_ind = cf.reject_extreme_values(z_nofv_nonan)
    t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
    y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
    z_nofv_nonan_noev = z_nofv_nonan[ev_ind]
    d_nofv_nonan_noev = d_nofv_nonan[ev_ind]
    print(len(z_nofv_nonan) - len(z_nofv_nonan_noev), ' Extreme Values', '|1e7|')

    # reject values outside global ranges:
    global_min, global_max = cf.get_global_ranges(r, v)
    if isinstance(global_min, (int, float)) and isinstance(global_max, (int, float)):
        gr_ind = cf.reject_global_ranges(z_nofv_nonan_noev, global_min, global_max)
        dtime = t_nofv_nonan_noev[gr_ind]
        zpressure = y_nofv_nonan_noev[gr_ind]
        ndata = z_nofv_nonan_noev[gr_ind]
        ndeploy = d_nofv_nonan_noev[gr_ind]
    else:
        gr_ind = []
        dtime = t_nofv_nonan_noev
        zpressure = y_nofv_nonan_noev
        ndata = z_nofv_nonan_noev
        ndeploy = d_nofv_nonan_noev

    print('{} global ranges [{} - {}]'.format(len(ndata) - len(z_nofv_nonan_noev), global_min, global_max))

    return dtime, zpressure, ndata, ndeploy


def reject_timestamps_data_portal(subsite, r, tt, yy, zz, dd):

    dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
    drn = dr.loc[dr.type == 'exclusion']

    if len(drn) != 0:
        subsite_node = '-'.join((subsite, r.split('-')[1]))
        drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]
        if len(drne['reference_designator']) != 0:
            t_ex = tt
            y_ex = yy
            z_ex = zz
            d_ex = dd
            for ij, row in drne.iterrows():
                sdate = cf.format_dates(row.start_date)
                edate = cf.format_dates(row.end_date)
                ts = np.datetime64(sdate)
                te = np.datetime64(edate)
                if t_ex.max() < ts:
                    continue
                elif t_ex.min() > te:
                    continue
                else:
                    ind = np.where((t_ex < ts) | (t_ex > te), True, False)
                    if len(ind) != 0:
                        t_ex = t_ex[ind]
                        z_ex = z_ex[ind]
                        y_ex = y_ex[ind]
                        d_ex = d_ex[ind]
                        print('Portal: excluding {} timestamps in [{} - {}]'.format(len(ind), sdate, edate))

    return t_ex, y_ex, z_ex, d_ex

def reject_timestamps_from_stat_analysis(Dpath, deployment, var, tt, yy, zz, dd):

    onlyfiles = []
    for item in os.listdir(Dpath):
        if not item.startswith('.') and os.path.isfile(os.path.join(Dpath, item)):
            if deployment in item:
                onlyfiles.append(join(Dpath, item))

    dre = pd.DataFrame()
    for nn in onlyfiles:
        dr = pd.read_csv(nn)
        dre = dre.append(dr, ignore_index=True)

    drn = dre.loc[dre['Unnamed: 0'] == var]
    list_time = []
    for itime in drn.time_to_exclude:
        ntime = itime.split(', ')
        list_time.extend(ntime)

    u_time_list = np.unique(list_time)
    if len(u_time_list) != 0:
        tt, yy, zz, dd = reject_suspect_data(tt, yy, zz, dd, u_time_list)

    return tt, yy, zz, dd

def reject_data_in_depth_range(tt, yy, zz, dd, zdbar):
    if zdbar is not None:
        y_ind = y_portal < zdbar
        if len(y_ind) != 0:
            tt = tt[y_ind]
            yy = yy[y_ind]
            zz = zz[y_ind]
            dd = dd[y_ind]
    return tt, yy, zz, dd


def reject_suspect_data(t, y, z, d, timestamps):

    data = pd.DataFrame({'yy': y, 'zz': z, 'dd': d, 'tt': t}, index=t)
    dtime = [(np.datetime64(pd.to_datetime(row))) for row in timestamps]

    if pd.to_datetime(t.max()) > pd.to_datetime(min(dtime)) or pd.to_datetime(t.min()) < pd.to_datetime(max(dtime)):
        ind = np.where((dtime >= t.min()) & (dtime <= t.max()))
        if len(dtime) - len(ind[0]) > 0:
            list_to_drop = [value for index, value in enumerate(dtime) if index in list(ind[0])]
            dtime = list_to_drop

        data = data.drop(dtime)

    return data['tt'], data['yy'].values, data['zz'].values, data['dd'].values
