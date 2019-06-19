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
        var_data = ds[var_id].values
        var_unit = ds[var_id].units
        var_name = ds[var_id].long_name
        var_fill = ds[var_id]._FillValue
    else:
        print('more than one matching name exist: ', var)

    return var_id, var_data, var_unit, var_name, var_fill


def compare_variable_attributes(fdatasets, r):
    vars_df = pd.DataFrame()
    for ii in range(len(fdatasets)):

        print('\n', fdatasets[ii].split('/')[-1])
        deployment = fdatasets[ii].split('/')[-1].split('_')[0].split('deployment')[-1]
        deployment = int(deployment)

        ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
        time = ds['time'].values

        dr_ms = '-'.join((str(deployment), ds.collection_method, ds.stream))

        '''
        variable list
        '''
        var_list = cf.notin_list(ds.data_vars.keys(), ['time', '_qc_'])

        z_id, z_data, z_unit, z_name, z_fill = cf.add_pressure_to_dictionary_of_sci_vars(ds)
        z_data, err_count_z = reject_err_data_1_dims(z_data, z_fill[0], r, z_name[0], n=5)
        df = pd.DataFrame({'vname_id':[z_id], 'units':[z_unit[0]], 'var_name':[z_name[0]]},index=[dr_ms])
        vars_df = vars_df.append(df)

        for vname in name_list:

            vname_id, vname_data, vname_unit, vname_name, vname_fill = get_variable_data(ds, var_list, vname)
            vname_data, err_count_vname = reject_err_data_1_dims(vname_data, vname_fill, r, vname_id, n=5)

            df = pd.DataFrame({'vname_id':[vname_id], 'units':[vname_unit], 'var_name':[vname_name]},index=[dr_ms])
            vars_df = vars_df.append(df)

    vars_df = vars_df.drop_duplicates()
    vars_df.to_csv('{}/{}_velocity_variables.csv'.format(sDir, r), index=True)

    return vars_df

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

def sci_var_long_names(refdes):
    # get science variable long names from data files
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
        vars_df.insert(loc=len(vars_df.columns), column='values', value=vars_df['vname_id'])
        for ii in range(len(fdatasets)):
            ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
            tD = ds['time'].values
            print(len(tD))
            dD = np.unique(ds['deployment'].values)
            for vv in range(len(vars_df['vname_id'])):
                vD = vars_df[vars_df['test-0'].values == vars_df['vname_id'].values[vv]]
                vD.insert(loc=len(vD.columns), column='v-insert', value=[ds[vars_df['vname_id'].values[vv]].values])

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    preferred_only = 'yes'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'

    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191340-CE06ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191211-CE06ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191157-CE06ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_instrument_recovered/catalog.html']

    name_list = ['upward_velocity', 'eastward_velocity', 'northward_velocity', 'roll', 'pitch']

main(sDir, url_list, preferred_only, name_list)