#!/usr/bin/env python
import os
import itertools
import pandas as pd
import xarray as xr
import numpy as np
import functions.common as cf
import matplotlib
from matplotlib import pyplot
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
from matplotlib.ticker import MaxNLocator


def get_variable_data(ds, var_list, keyword):
    var = [var for var in var_list if keyword in var]
    if len(var) == 1:
        var_id = var[0]
        var_unit = ds[var_id].units
        var_name = ds[var_id].long_name
    else:
        print('more than one matching name exist: ', var)

    return var_id, var_unit, var_name


def compare_variable_attributes(fdatasets, r):
    vars_df = pd.DataFrame()
    for ii in range(len(fdatasets)):

        print('\n', fdatasets[ii].split('/')[-1])
        deployment = fdatasets[ii].split('/')[-1].split('_')[0].split('deployment')[-1]
        deployment = int(deployment)

        ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
        time = ds['time'].values

        dr_dp = '-'.join((str(deployment))) #,ds.collection_method, ds.stream

        '''
        variable list
        '''
        var_list = cf.notin_list(ds.data_vars.keys(), ['time', '_qc_'])

        z_id, z_data, z_unit, z_name, z_fill = cf.add_pressure_to_dictionary_of_sci_vars(ds)
        df = pd.DataFrame({'vname_id':[z_id], 'units':[z_unit[0]], 'var_name':[z_name[0]]},index=[dr_dp])
        vars_df = vars_df.append(df)

        for vname in name_list:

            vname_id, vname_unit, vname_name = get_variable_data(ds, var_list, vname)

            df = pd.DataFrame({'vname_id':[vname_id], 'units':[vname_unit], 'var_name':[vname_name]},index=[dr_dp])
            vars_df = vars_df.append(df)

    vars_df = vars_df.drop_duplicates()
    vars_df.to_csv('{}/{}_velocity_variables.csv'.format(sDir, r), index=True)

    return vars_df

def append_data(vars_df, fdatasets):
    vars_df.insert(loc=len(vars_df.columns), column='values', value=vars_df['vname_id'])
    vars_df.insert(loc=len(vars_df.columns), column='fill_values', value=vars_df.index)
    vars_df.insert(loc=len(vars_df.columns), column='preferred_methods_streams', value=vars_df.index)

    columns = ['vname_id', 'units', 'var_name', 'fill_values', 'values']

    df_time = pd.DataFrame(np.array([['time', '', 'Time', 'time']]), columns=columns)
    vars_df = df_time.append(vars_df)
    df_deployment = pd.DataFrame(np.array([['deployment', '', 'Deployment Number', 'deployment']]), columns=columns)
    vars_df = df_deployment.append(vars_df)
    vars_df.index = vars_df['vname_id']

    all_in_one_df = pd.DataFrame()
    add_to_df = pd.DataFrame()

    for ii in [0, 1]:  # range(len(fdatasets)):
        print('\n', fdatasets[ii].split('/')[-1])
        ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
        dr_ms = '-'.join((ds.collection_method, ds.stream))

        # get data:
        tD = ds['time'].values
        print(len(tD))
        dD = np.unique(ds['deployment'].values)

        for vv in range(len(vars_df['vname_id'])):

            if ii == 0:
                vD = vars_df[vars_df['values'].values == vars_df['vname_id'].values[vv]]
                var = ds[vD['vname_id'][0]].values
                if len(np.unique(var)) == 1:
                    var = np.unique(var)
                fv = ds[vD['vname_id'][0]]._FillValue
                vD.at[vD.index.values[0], 'values'] = var
                vD.at[vD.index.values[0], 'fill_values'] = fv
                vD.at[vD.index.values[0], 'preferred_methods_streams'] = dr_ms
            else:
                vD = add_to_df[add_to_df['vname_id'].values == vars_df['vname_id'].values[vv]]
                var = ds[vD['vname_id'][0]].values
                if len(np.unique(var)) == 1:
                    var = np.unique(va)
                fv = ds[vD['vname_id'][0]]._FillValue
                vD.at[vD.index.values[0], 'values'] = np.append(vD['values'].values[0], var)
                vD.at[vD.index.values[0], 'fill_values'] = np.append(vD['fill_values'].values[0], fv)
                vD.at[vD.index.values[0], 'preferred_methods_streams'] = np.append(vD['preferred_methods_streams'].values[0], dr_ms)

            add_to_df = add_to_df.append(vD)

    all_in_one_df = add_to_df[len(add_to_df) - len(vars_df['vname_id']):len(add_to_df)]

    return all_in_one_df

def reject_err_data_1_dims(y, y_fill, r, sv, n=None):
    n_nan = np.sum(np.isnan(y)) # count nans in data
    n_nan = n_nan.item()
    y = np.where(y != y_fill, y, np.nan) # replace fill_values by nans in data
    y = np.where(y != -9999, y, np.nan) # replace -9999 by nans in data
    n_fv = np.sum(np.isnan(y)) - n_nan# re-count nans in data
    n_fv = n_fv.item()
    y = np.where(y > -1e10, y, np.nan) # replace extreme values by nans in data
    y = np.where(y < 1e10, y, np.nan)
    n_ev = np.sum(np.isnan(y)) - n_fv - n_nan # re-count nans in data
    n_ev = n_ev.item()

    g_min, g_max = cf.get_global_ranges(r, sv) # get global ranges:
    if g_min and g_max:
        y = np.where(y >= g_min, y, np.nan) # replace extreme values by nans in data
        y = np.where(y <= g_max, y, np.nan)
        n_grange = np.sum(np.isnan(y)) - n_ev - n_fv - n_nan # re-count nans in data
        n_grange = n_grange.item()
    else:
        n_grange = np.nan

    stdev = np.nanstd(y)
    if stdev > 0.0:
        y = np.where(abs(y - np.nanmean(y)) < n * stdev, y, np.nan) # replace 5 STD by nans in data
        n_std = np.sum(np.isnan(y)) - ~np.isnan(n_grange) - n_ev - n_fv - n_nan # re-count nans in data
        n_std = n_std.item()

    err_count = pd.DataFrame({'n_nan':[n_nan],
                             'n_fv':[n_fv],
                             'n_ev':[n_ev],
                             'n_grange':[n_grange],
                             'g_min':[g_min],
                             'g_max':[g_max],
                             'n_std':[n_std]}, index=[0])
    return  y, err_count


def main(sDir, url_list, preferred_only, name_list):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = uu.split(rd + '-')[1].split('/')[0]
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        subsite = r.split('-')[0]
        array = subsite[0:2]

        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)

        datasets = list(itertools.chain(*datasets))

        if preferred_only == 'yes':

            ps_df, n_streams = cf.get_preferred_stream_info(r)

            fdatasets = []
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

        main_sensor = r.split('-')[-1]
        fdatasets = cf.filter_collocated_instruments(main_sensor, fdatasets)

        vars_df = compare_variable_attributes(fdatasets, r)
        if len(np.unique(vars_df.index.values)) == 1 and len(vars_df['vname_id']) == len(name_list)+1:
            print('Pass: variables exist in all files')
        else:
            print('Fail: variables differ in between files')

        #all_in_one_df = append_data(vars_df, fdatasets)

        vars_df.insert(loc=len(vars_df.columns), column='values', value=vars_df['vname_id'])
        vars_df.insert(loc=len(vars_df.columns), column='fill_values', value=vars_df.index)
        vars_df.insert(loc=0, column='preferred_methods_streams', value=vars_df.index)

        columns = ['preferred_methods_streams','vname_id', 'units', 'var_name', 'fill_values', 'values']

        df_time = pd.DataFrame(np.array([['', 'time', '', 'Time','', 'time']]), columns=columns)
        vars_df = df_time.append(vars_df)
        df_deployment = pd.DataFrame(np.array([['','deployment', '', 'Deployment Number','', 'deployment']]), columns=columns)
        vars_df = df_deployment.append(vars_df)
        vars_df.index = vars_df['vname_id']

        all_in_one_df = pd.DataFrame()
        add_to_df = pd.DataFrame()

        for ii in [0, 1]:  # range(len(fdatasets)):
            print('\n', fdatasets[ii].split('/')[-1])
            ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
            dr_ms = '-'.join((ds.collection_method, ds.stream))

            # get data:
            tD = ds['time'].values
            print(len(tD))
            dD = np.unique(ds['deployment'].values)

            for vv in range(len(vars_df['vname_id'])):

                if ii == 0:
                    vD = vars_df[vars_df['values'].values == vars_df['vname_id'].values[vv]]
                    var = ds[vD['vname_id'][0]].values
                    if len(np.unique(var)) == 1:
                        var = np.unique(var)
                    try:
                        fv = ds[vD['vname_id'][0]]._FillValue
                    except AttributeError:
                        fv =''

                    vD.at[vD.index.values[0], 'values'] = var
                    vD.at[vD.index.values[0], 'fill_values'] = np.unique(fv)
                    vD.at[vD.index.values[0], 'preferred_methods_streams'] = np.unique(dr_ms)
                else:
                    vD = add_to_df[add_to_df['vname_id'].values == vars_df['vname_id'].values[vv]]
                    var = ds[vD['vname_id'][0]].values
                    if len(np.unique(var)) == 1:
                        var = np.unique(var)
                    try:
                        fv = ds[vD['vname_id'][0]]._FillValue
                    except AttributeError:
                        fv = ''
                    vD.at[vD.index.values[0], 'values'] = np.append(vD['values'].values[0], var)
                    vD.at[vD.index.values[0], 'fill_values'] = np.append(vD['fill_values'].values[0], fv)
                    vD.at[vD.index.values[0], 'preferred_methods_streams'] = np.append(
                        vD['preferred_methods_streams'].values[0], dr_ms)

                add_to_df = add_to_df.append(vD)

        all_in_one_df = add_to_df[len(add_to_df) - len(vars_df['vname_id']):len(add_to_df)]



        # vars_df.insert(loc=len(vars_df.columns), column='values', value=vars_df['vname_id'])
        #         #
        #         # columns = ['vname_id', 'units', 'var_name', 'values']
        #         #
        #         # df_time = pd.DataFrame(np.array([['time', '', 'Time', 'time']]), columns=columns)
        #         # vars_df = df_time.append(vars_df)
        #         # df_deployment = pd.DataFrame(np.array([['deployment', '', 'Deployment Number', 'deployment']]),columns=columns)
        #         # vars_df = df_deployment.append(vars_df)
        #         # vars_df.index = vars_df['vname_id']
        #         #
        #         # all_in_one_df = pd.DataFrame()
        #         # add_to_df = pd.DataFrame()
        #         #
        #         # for ii in [0,1]:#range(len(fdatasets)):
        #         #     print('\n', fdatasets[ii].split('/')[-1])
        #         #     ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
        #         #     dr_ms = '-'.join((ds.collection_method, ds.stream))
        #         #     # get data:
        #         #     tD = ds['time'].values
        #         #     print(len(tD))
        #         #     dD = np.unique(ds['deployment'].values)
        #         #
        #         #     for vv in range(len(vars_df['vname_id'])):
        #         #
        #         #         if ii == 0:
        #         #             vD = vars_df[vars_df['values'].values == vars_df['vname_id'].values[vv]]
        #         #             var = ds[vD['vname_id'][0]].values
        #         #             if len(np.unique(var)) == 1:
        #         #                 var = np.unique(var)
        #         #             vD.at[vD.index.values[0], 'values'] = var
        #         #         else:
        #         #             vD = add_to_df[add_to_df['vname_id'].values == vars_df['vname_id'].values[vv]]
        #         #             var = ds[vD['vname_id'][0]].values
        #         #             if len(np.unique(var)) == 1:
        #         #                 var = np.unique(var)
        #         #             vD.at[vD.index.values[0], 'values'] = np.append(vD['values'].values[0], var)
        #         #
        #         #         add_to_df = add_to_df.append(vD)
        #         #
        #         # all_in_one_df = add_to_df[len(add_to_df) - len(vars_df['vname_id']):len(add_to_df)]

        df_roll = all_in_one_df[all_in_one_df.vname_id.str.contains('roll') == True]
        df_pitch = all_in_one_df[all_in_one_df.vname_id.str.contains('pitch') == True]

        # select data with pitch or roll less than 20 degrees
        ind = np.logical_and(df_roll['values'].values[0] < 200, df_pitch['values'].values[0] < 200)

        roll = df_roll['values'].values[0]
        c_roll = roll[ind]
        mean_roll, max_roll, min_roll, std_roll = np.nanmean(c_roll), np.nanmax(c_roll), np.nanmin(c_roll), np.nanstd(c_roll)


        df_U = all_in_one_df[all_in_one_df.vname_id.str.contains('eastward') == True]
        u = df_U['values'].values[0]
        uu = u[ind]
        mean_u, max_u, min_u, std_u = np.nanmean(uu), np.nanmax(uu), np.nanmin(uu), np.nanstd(uu)

        df_V = all_in_one_df[all_in_one_df.vname_id.str.contains('northward') == True]
        v = df_V['values'].values[0]
        vv = v[ind]
        mean_v, max_v, min_v, std_v = np.nanmean(vv), np.nanmax(vv), np.nanmin(vv), np.nanstd(vv)

        df_W = all_in_one_df[all_in_one_df.vname_id.str.contains('upward') == True]
        w = df_W['values'].values[0]
        ww = w[ind]
        mean_w, max_w, min_w, std_w = np.nanmean(ww), np.nanmax(ww), np.nanmin(ww), np.nanstd(ww)

        rows =[]
        columns = []
        common_stream_name,preferred_methods_streams,deployments,[[long_name,units]],t0,t1,[[fill_value,global_ranges,n_all,n_nans,n_fillvalues,n_grange]],define_stdev,n_outliers,n_stats,mean,min,max,stdev
        rows.append([m, list(np.unique(pms)), deployments, sv, lunits, t0, t1, fv_lst, [g_min, g_max],
                                         n_all, [round(ipress_min, 2), round(ipress_max, 2)], n_excluded, n_nan, n_fv, n_grange,
                                 sd_calc, num_outliers, n_stats, mean, vmin, vmax, sd, note])

    print('Stop')


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    preferred_only = 'yes'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'

    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191340-CE06ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191211-CE06ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191157-CE06ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_instrument_recovered/catalog.html']

    name_list = ['upward_velocity', 'eastward_velocity', 'northward_velocity', 'roll', 'pitch']

main(sDir, url_list, preferred_only, name_list)