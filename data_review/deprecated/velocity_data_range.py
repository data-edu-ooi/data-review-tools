#!/usr/bin/env python
# import os
import itertools
import pandas as pd
import xarray as xr
import numpy as np
import functions.common as cf
# import matplotlib
# from matplotlib import pyplot
# import datetime
# import matplotlib.dates as mdates
# import matplotlib.ticker as ticker
# from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
# from matplotlib.ticker import MaxNLocator


# exist in main function
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
    if stdev != 0:
        if n is not None:
            y = np.where(abs(y - np.nanmean(y)) < n * stdev, y, np.nan) # replace 5 STD by nans in data
            n_std = np.sum(np.isnan(y)) - ~np.isnan(n_grange) - n_ev - n_fv - n_nan # re-count nans in data
            n_std = n_std.item()
        else:
            n_std = np.nan

    err_count = pd.DataFrame({'global_ranges':[[g_min,g_max]], 'n_nans':[n_nan], 'n_fillvalues':[n_fv],
                              'n_extremvalues':[n_ev], 'n_grange':[n_grange], 'define_stdev':[n],'n_outliers':[n_std],
                              'n_stats':[np.sum(~np.isnan(y))]}, index=[0])
    return y, err_count


# exist in sub_function
def get_variable_data(ds, var_list, keyword):
    var = [var for var in var_list if keyword in var]
    if len(var) == 1:
        var_id = var[0]
        var_unit = ds[var_id].units
        var_name = ds[var_id].long_name
        try:
            var_fv = ds[var_id]._FillValue
        except AttributeError:
            var_fv = ''
    else:
        print('more than one matching name exist: ', var)

    return var_id, var_unit, var_name, var_fv


# exist in main function
def compare_variable_attributes(fdatasets, r, name_list, sDir):
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
        df = pd.DataFrame({'var_id':[z_id], 'units':[z_unit[0]], 'long_name':[z_name[0]], 'fill_values':[z_fill[0]]},index=[dr_dp])
        vars_df = vars_df.append(df)

        for vname in name_list:

            vname_id, vname_unit, vname_name, vname_fv = get_variable_data(ds, var_list, vname)

            df = pd.DataFrame({'var_id':[vname_id], 'units':[vname_unit],
                               'long_name':[vname_name], 'fill_values':[str(vname_fv)]},index=[dr_dp])
            vars_df = vars_df.append(df)

    vars_df = vars_df.drop_duplicates()
    vars_df.to_csv('{}/{}_velocity_variables.csv'.format(sDir, r), index=True)

    return vars_df


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

        vars_df = compare_variable_attributes(fdatasets, r, name_list, sDir)
        if len(np.unique(vars_df.index.values)) == 1 and len(vars_df['var_id']) == len(name_list)+1:
            print('Pass: variables exist in all files')
        else:
            print('Fail: variables differ in between files')

        # all_in_one_df = append_data(vars_df, fdatasets)

        vars_df.insert(loc=len(vars_df.columns), column='values', value=vars_df['var_id'])
        vars_df.insert(loc=len(vars_df.columns), column='t0', value=vars_df['var_id'])
        vars_df.insert(loc=len(vars_df.columns), column='t1', value=vars_df['var_id'])
        vars_df.insert(loc=len(vars_df.columns), column='deployments', value=vars_df['var_id'])
        vars_df.insert(loc=len(vars_df.columns), column='preferred_methods_streams', value=vars_df['var_id'])
        vars_df.insert(loc=len(vars_df.columns), column='common_stream_name', value=vars_df['var_id'])

        vars_df.index = vars_df['var_id']

        add_to_df = pd.DataFrame()

        for ii in range(len(fdatasets)):  # [4, 5]
            print('\n', fdatasets[ii].split('/')[-1])
            ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
            var_ms = '-'.join((ds.collection_method, ds.stream))
            var_s = ds.stream
            var_d = ds['deployment'].values

            time_d = ds['time'].values

            for vv in range(len(vars_df['var_id'])):
                print(vars_df['var_id'].values[vv])
                if ii == 0:  #if ii == 4
                    vD = vars_df[vars_df['values'].values == vars_df['var_id'].values[vv]]
                    data_v = ds[vD['var_id'][0]].values
                    if len(np.unique(data_v)) == 1:
                        data_v = np.unique(data_v)

                    vD.at[vD.index.values[0], 'values'] = data_v
                    vD.at[vD.index.values[0], 'deployments'] = np.unique(var_d)
                    vD.at[vD.index.values[0], 't0'] = time_d[0]
                    vD.at[vD.index.values[0], 't1'] = time_d[len(time_d)-1]
                    vD.at[vD.index.values[0], 'preferred_methods_streams'] = np.unique(var_ms)
                    vD.at[vD.index.values[0], 'common_stream_name'] = np.unique(var_s)
                else:
                    vD = add_to_df[add_to_df['var_id'].values == vars_df['var_id'].values[vv]]
                    data_v = ds[vD['var_id'][0]].values
                    if len(np.unique(data_v)) == 1:
                        data_v = np.unique(data_v)

                    vD.at[vD.index.values[0], 'values'] = np.append(vD['values'].values[0], data_v)
                    vD.at[vD.index.values[0], 'deployments'] = np.unique(np.append(vD['deployments'].values[0], var_d))
                    vD.at[vD.index.values[0], 't0'] = np.unique(np.append(vD['t0'].values[0],time_d[0]))
                    vD.at[vD.index.values[0], 't1'] = np.unique(np.append(vD['t1'].values[0],time_d[len(time_d) - 1]))
                    vD.at[vD.index.values[0], 'preferred_methods_streams'] = np.unique(np.append(
                        vD['preferred_methods_streams'].values[0], var_ms))
                    vD.at[vD.index.values[0], 'common_stream_name'] = np.unique(np.append(vD['common_stream_name'].values[0], var_s))

                add_to_df = add_to_df.append(vD)

        all_in_one_df = add_to_df[len(add_to_df) - len(vars_df['var_id']):len(add_to_df)]

        df_roll = all_in_one_df[all_in_one_df.var_id.str.contains('roll') == True]
        df_pitch = all_in_one_df[all_in_one_df.var_id.str.contains('pitch') == True]

        # define indices to select data with pitch or roll less than 20 degrees
        ind = np.logical_and(df_roll['values'].values[0] < 200, df_pitch['values'].values[0] < 200)

        df_F = pd.DataFrame()
        for keyword in ['eastward', 'northward', 'upward', 'pressure']: #
            df_k = all_in_one_df[all_in_one_df.var_id.str.contains(keyword) == True]
            print(keyword, len(df_k['t0'].values), len(df_k['t1'].values))
            if len(df_k['t1'].values) > 1:
                df_k.at[df_k.index.values[0], 't1'] = max(df_k['t1'].values[0])
            if len(df_k['t0'].values) > 1:
                df_k.at[df_k.index.values[0], 't0'] = min(df_k['t0'].values[0])

            u = df_k['values'].values[0]
            uu = u[ind]
            uu_fv = float(df_k['fill_values'].values[0])
            uu_id = df_k['var_id'].values[0]
            u_wo_err, u_err = reject_err_data_1_dims(uu, uu_fv, r, uu_id, n=None)

            u_err.insert(loc=0, column='n_all', value=len(u))
            u_err.insert(loc=1, column='n_pitchroll_err', value=len(u) - len(uu))
            u_err.insert(loc=len(u_err.columns), column='mean', value=np.nanmean(u_wo_err))
            u_err.insert(loc=len(u_err.columns), column='min', value=np.nanmin(u_wo_err))
            u_err.insert(loc=len(u_err.columns), column='max', value=np.nanmax(u_wo_err))
            u_err.insert(loc=len(u_err.columns), column='stdev', value=np.nanstd(u_wo_err))
            for name in u_err.columns:
                df_k.insert(loc=len(df_k.columns), column=name, value=u_err[name].values)

            df_F = df_F.append(df_k)

        columns = ['common_stream_name', 'preferred_methods_streams', 'deployments', 'long_name', 'units',
                   't0', 't1', 'fill_values', 'global_ranges', 'n_all', 'n_pitchroll_err', 'n_nans', 'n_fillvalues',
                   'n_extremvalues', 'n_grange', 'define_stdev', 'n_outliers', 'n_stats', 'mean', 'min', 'max', 'stdev']

        df_F.to_csv(sDir+array+'/'+subsite+'/'+r+'_data_ranges.csv', columns=columns, index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    preferred_only = 'yes'
    #sDir = '/Users/leila/Documents/NSFEduSupport/github/data-review-tools/data_review/data_ranges/'
    sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/data_ranges'

    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191340-CE06ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191211-CE06ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191157-CE06ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_instrument_recovered/catalog.html']

    name_list = ['upward_velocity', 'eastward_velocity', 'northward_velocity', 'roll', 'pitch']

    main(sDir, url_list, preferred_only, name_list)
