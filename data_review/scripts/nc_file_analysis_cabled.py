#!/usr/bin/env python
"""
Created on Sep 12 2019 by Lori Garzio
@brief Automated analysis of cabled .nc files from a THREDDs server. Provides a json output by reference designator.
sDir: location to save summary output
url: list of THREDDs urls containing .nc files to analyze.
"""

import os
import xarray as xr
import pandas as pd
import re
import numpy as np
from collections import OrderedDict
import json
import datetime as dt
from datetime import timedelta
import netCDF4 as nc
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd


def compare_lists(list1, list2):
    match = []
    unmatch = []
    for i in list1:
        if i in list2:
            match.append(i)
        else:
            unmatch.append(i)
    return match, unmatch


def eliminate_common_variables(list):
    # time is in this list because it doesn't show up as a variable in an xarray ds
    common = ['quality_flag', 'provenance', 'id', 'deployment', 'obs', 'lat', 'lon']
    regex = re.compile(r'\b(?:%s)\b' % '|'.join(common))
    list = [s for s in list if not regex.search(s)]
    return list


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def insert_into_dict(d, key, value):
    if key not in d:
        d[key] = [value]
    else:
        d[key].append(value)
    return d


def main(sDir, url_list, deployment_num):
    reviewlist = pd.read_csv(
        'https://raw.githubusercontent.com/ooi-data-lab/data-review-prep/master/review_list/data_review_list.csv')

    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    json_file_list = []
    for r in rd_list:
        dependencies = []
        print('\n{}'.format(r))
        data = OrderedDict(deployments=OrderedDict())
        save_dir = os.path.join(sDir, r.split('-')[0], r)
        cf.create_dir(save_dir)

        # Deployment location test
        deploy_loc_test = cf.deploy_location_check(r)
        data['location_comparison'] = deploy_loc_test

        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            catalog_rms = '-'.join((r, splitter[-2], splitter[-1]))

            # complete the analysis by reference designator
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])

                # check for the OOI 1.0 datasets for review
                rl_filtered = reviewlist.loc[
                    (reviewlist['Reference Designator'] == r) & (reviewlist['status'] == 'for review')]
                review_deployments = rl_filtered['deploymentNumber'].tolist()
                review_deployments_int = ['deployment%04d' % int(x) for x in review_deployments]

                for rev_dep in review_deployments_int:
                    if deployment_num is not None:
                        if int(rev_dep[-4:]) is not deployment_num:
                            print('\nskipping {}'.format(rev_dep))
                            continue

                    rdatasets = [s for s in udatasets if rev_dep in s]
                    rdatasets.sort()
                    if len(rdatasets) > 0:
                        datasets = []
                        for dss in rdatasets:  # filter out collocated data files
                            if catalog_rms == dss.split('/')[-1].split('_20')[0][15:]:
                                datasets.append(dss)
                            else:
                                drd = dss.split('/')[-1].split('_20')[0][15:42]
                                if drd not in dependencies and drd != r:
                                    dependencies.append(drd)

                        notes = []
                        time_ascending = ''
                        sci_vars_dict = {}
                        #datasets = datasets[0:2]  #### for testing
                        for i in range(len(datasets)):
                            ds = xr.open_dataset(datasets[i], mask_and_scale=False)
                            ds = ds.swap_dims({'obs': 'time'})
                            print('\nAppending data from {}: file {} of {}'.format(rev_dep, i+1, len(datasets)))

                            # when opening multiple datasets, don't check that the timestamps are in ascending order
                            time_ascending = 'not_tested'

                            if i == 0:
                                fname, subsite, refdes, method, data_stream, deployment = cf.nc_attributes(datasets[0])
                                fname = fname.split('_20')[0]

                                # Get info from the data review database
                                dr_data = cf.refdes_datareview_json(refdes)
                                stream_vars = cf.return_stream_vars(data_stream)
                                sci_vars = cf.return_science_vars(data_stream)
                                node = refdes.split('-')[1]
                                if 'cspp' in data_stream or 'WFP' in node:
                                    sci_vars.append('int_ctd_pressure')

                                # Add pressure to the list of science variables
                                press = pf.pressure_var(ds, list(ds.coords.keys()))
                                if press is None:
                                    press = pf.pressure_var(ds, list(ds.data_vars.keys()))
                                if press is not None:
                                    sci_vars.append(press)
                                sci_vars.append('time')
                                sci_vars = list(np.unique(sci_vars))
                                if 'ADCP' in r:
                                    sci_vars = [x for x in sci_vars if 'beam' not in x]

                                for sci_var in sci_vars:
                                    if sci_var == 'time':
                                        sci_vars_dict.update(
                                            {sci_var: dict(values=np.array([], dtype=np.datetime64), units=[], fv=[])})
                                    else:
                                        sci_vars_dict.update({sci_var: dict(values=np.array([]), units=[], fv=[])})

                                deploy_info = get_deployment_information(dr_data, int(deployment[-4:]))

                                # Grab deployment Variables
                                deploy_start = str(deploy_info['start_date'])
                                deploy_stop = str(deploy_info['stop_date'])
                                deploy_lon = deploy_info['longitude']
                                deploy_lat = deploy_info['latitude']
                                deploy_depth = deploy_info['deployment_depth']

                                # Calculate days deployed
                                if deploy_stop != 'None':
                                    r_deploy_start = pd.to_datetime(deploy_start).replace(hour=0, minute=0, second=0)
                                    if deploy_stop.split('T')[1] == '00:00:00':
                                        r_deploy_stop = pd.to_datetime(deploy_stop)
                                    else:
                                        r_deploy_stop = (pd.to_datetime(deploy_stop) + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                                    n_days_deployed = (r_deploy_stop - r_deploy_start).days
                                else:
                                    n_days_deployed = None

                                # Add reference designator to dictionary
                                try:
                                    data['refdes']
                                except KeyError:
                                    data['refdes'] = refdes

                            # append data for the deployment into a dictionary
                            for s_v in sci_vars_dict.keys():
                                vv = ds[s_v]
                                try:
                                    if vv.units not in sci_vars_dict[s_v]['units']:
                                        sci_vars_dict[s_v]['units'].append(vv.units)
                                except AttributeError:
                                    print('')
                                try:
                                    if vv._FillValue not in sci_vars_dict[s_v]['fv']:
                                        sci_vars_dict[s_v]['fv'].append(vv._FillValue)
                                except AttributeError:
                                    print('')
                                if len(vv.dims) == 1:
                                    if s_v in ['wavelength_a', 'wavelength_c']:
                                        # if the array is not same as the array that was already appended for these
                                        # two OPTAA variables, append. if it's already there, don't append
                                        if np.sum(vv.values == sci_vars_dict[s_v]['values']) != len(vv.values):
                                            sci_vars_dict[s_v]['values'] = np.append(sci_vars_dict[s_v]['values'],
                                                                                     vv.values)
                                    else:
                                        sci_vars_dict[s_v]['values'] = np.append(sci_vars_dict[s_v]['values'], vv.values)

                                elif len(vv.dims) == 2:  # appending 2D datasets
                                    vD = vv.values.T
                                    if len(sci_vars_dict[s_v]['values']) == 0:
                                        sci_vars_dict[s_v]['values'] = vD
                                    else:
                                        sci_vars_dict[s_v]['values'] = np.concatenate((sci_vars_dict[s_v]['values'], vD), axis=1)

                        deployments = data['deployments'].keys()
                        data_start = pd.to_datetime(min(sci_vars_dict['time']['values'])).strftime('%Y-%m-%dT%H:%M:%S')
                        data_stop = pd.to_datetime(max(sci_vars_dict['time']['values'])).strftime('%Y-%m-%dT%H:%M:%S')

                        # Add deployment and info to dictionary and initialize delivery method sub-dictionary
                        if deployment not in deployments:
                            data['deployments'][deployment] = OrderedDict(deploy_start=deploy_start,
                                                                          deploy_stop=deploy_stop,
                                                                          n_days_deployed=n_days_deployed,
                                                                          lon=deploy_lon,
                                                                          lat=deploy_lat,
                                                                          deploy_depth=deploy_depth,
                                                                          method=OrderedDict())

                        # Add delivery methods to dictionary and initialize stream sub-dictionary
                        methods = data['deployments'][deployment]['method'].keys()
                        if method not in methods:
                            data['deployments'][deployment]['method'][method] = OrderedDict(
                                stream=OrderedDict())

                        # Add streams to dictionary and initialize file sub-dictionary
                        streams = data['deployments'][deployment]['method'][method]['stream'].keys()

                        if data_stream not in streams:
                            data['deployments'][deployment]['method'][method]['stream'][
                                data_stream] = OrderedDict(file=OrderedDict())

                        # Get a list of data gaps >1 day
                        time_df = pd.DataFrame(sci_vars_dict['time']['values'], columns=['time'])
                        time_df = time_df.sort_values(by=['time'])
                        gap_list = cf.timestamp_gap_test(time_df)

                        # Calculate the sampling rate to the nearest second
                        time_df['diff'] = time_df['time'].diff().astype('timedelta64[s]')
                        rates_df = time_df.groupby(['diff']).agg(['count'])
                        n_diff_calc = len(time_df) - 1
                        rates = dict(n_unique_rates=len(rates_df), common_sampling_rates=dict())
                        for i, row in rates_df.iterrows():
                            percent = (float(row['time']['count']) / float(n_diff_calc))
                            if percent > 0.1:
                                rates['common_sampling_rates'].update({int(i): '{:.2%}'.format(percent)})

                        sampling_rt_sec = None
                        for k, v in rates['common_sampling_rates'].items():
                            if float(v.strip('%')) > 50.00:
                                sampling_rt_sec = k

                        if not sampling_rt_sec:
                            sampling_rt_sec = 'no consistent sampling rate: {}'.format(rates['common_sampling_rates'])

                        # Don't do : Check that the timestamps in the file are unique
                        time_test = ''

                        # Count the number of days for which there is at least 1 timestamp
                        n_days = len(np.unique(sci_vars_dict['time']['values'].astype('datetime64[D]')))

                        # Compare variables in file to variables in Data Review Database
                        ds_variables = list(ds.data_vars.keys()) + list(ds.coords.keys())
                        ds_variables = eliminate_common_variables(ds_variables)
                        ds_variables = [x for x in ds_variables if 'qc' not in x]
                        [_, unmatch1] = compare_lists(stream_vars, ds_variables)
                        [_, unmatch2] = compare_lists(ds_variables, stream_vars)

                        # calculate mean pressure from data, excluding outliers +/- 3 SD
                        try:
                            pressure = sci_vars_dict[press]
                            if len(pressure) > 1:
                                # reject NaNs
                                p_nonan = pressure['values'][~np.isnan(pressure['values'])]

                                # reject fill values
                                p_nonan_nofv = p_nonan[p_nonan != pressure['fv'][0]]

                                # reject data outside of global ranges
                                [pg_min, pg_max] = cf.get_global_ranges(r, press)
                                if pg_min is not None and pg_max is not None:
                                    pgr_ind = cf.reject_global_ranges(p_nonan_nofv, pg_min, pg_max)
                                    p_nonan_nofv_gr = p_nonan_nofv[pgr_ind]
                                else:
                                    p_nonan_nofv_gr = p_nonan_nofv

                                if (len(p_nonan_nofv_gr) > 0):
                                    [press_outliers, pressure_mean, _, pressure_max, _, _] = cf.variable_statistics(p_nonan_nofv_gr, 3)
                                    pressure_mean = round(pressure_mean, 2)
                                    pressure_max = round(pressure_max, 2)
                                else:
                                    press_outliers = None
                                    pressure_mean = None
                                    pressure_max = None
                                    if len(pressure) > 0 and len(p_nonan) == 0:
                                        notes.append('Pressure variable all NaNs')
                                    elif len(pressure) > 0 and len(p_nonan) > 0 and len(p_nonan_nofv) == 0:
                                        notes.append('Pressure variable all fill values')
                                    elif len(pressure) > 0 and len(p_nonan) > 0 and len(p_nonan_nofv) > 0 and len(p_nonan_nofv_gr) == 0:
                                        notes.append('Pressure variable outside of global ranges')

                            else:  # if there is only 1 data point
                                press_outliers = 0
                                pressure_mean = round(ds[press].values.tolist()[0], 2)
                                pressure_max = round(ds[press].values.tolist()[0], 2)

                            try:
                                pressure_units = pressure['units']
                            except AttributeError:
                                pressure_units = 'no units attribute for pressure'

                            if pressure_mean:
                                if 'SF' in node:
                                    pressure_compare = int(round(pressure_max))
                                else:
                                    pressure_compare = int(round(pressure_mean))

                                if pressure_units == '0.001 dbar':
                                    pressure_max = round((pressure_max / 1000), 2)
                                    pressure_mean = round((pressure_mean / 1000), 2)
                                    pressure_compare = round((pressure_compare / 1000), 2)
                                    notes.append('Pressure converted from 0.001 dbar to dbar for pressure comparison')

                                elif pressure_units == 'daPa':
                                    pressure_max = round((pressure_max / 1000), 2)
                                    pressure_mean = round((pressure_mean / 1000), 2)
                                    pressure_compare = round((pressure_compare / 1000), 2)
                                    notes.append('Pressure converted from daPa to dbar for pressure comparison')

                            else:
                                pressure_compare = None

                            if (not deploy_depth) or (not pressure_mean):
                                pressure_diff = None
                            else:
                                pressure_diff = pressure_compare - deploy_depth

                        except KeyError:
                            press = 'no seawater pressure in file'
                            pressure_diff = None
                            pressure_mean = None
                            pressure_max = None
                            pressure_compare = None
                            press_outliers = None
                            pressure_units = None

                        # Add files and info to dictionary
                        filenames = data['deployments'][deployment]['method'][method]['stream'][data_stream][
                            'file'].keys()

                        if fname not in filenames:
                            data['deployments'][deployment]['method'][method]['stream'][data_stream]['file'][
                                fname] = OrderedDict(
                                file_downloaded=pd.to_datetime(splitter[0][0:15]).strftime('%Y-%m-%dT%H:%M:%S'),
                                file_coordinates=list(ds.coords.keys()),
                                sampling_rate_seconds=sampling_rt_sec,
                                sampling_rate_details=rates,
                                data_start=data_start,
                                data_stop=data_stop,
                                time_gaps=gap_list,
                                unique_timestamps=time_test,
                                n_timestamps=len(sci_vars_dict['time']['values']),
                                n_days=n_days,
                                notes=notes,
                                ascending_timestamps=time_ascending,
                                pressure_comparison=dict(pressure_mean=pressure_mean, units=pressure_units,
                                                         num_outliers=press_outliers, diff=pressure_diff,
                                                         pressure_max=pressure_max, variable=press,
                                                         pressure_compare=pressure_compare),
                                vars_in_file=ds_variables,
                                vars_not_in_file=[x for x in unmatch1 if 'time' not in x],
                                vars_not_in_db=unmatch2,
                                sci_var_stats=OrderedDict())

                        # calculate statistics for science variables, excluding outliers +/- 5 SD
                        for sv in sci_vars_dict.keys():
                            if sv != 't_max':  # for ADCP
                                if sv != 'time':
                                    print(sv)
                                    var = sci_vars_dict[sv]
                                    vD = var['values']
                                    var_units = var['units']
                                    #if 'timedelta' not in str(vD.dtype):
                                    vnum_dims = len(np.shape(vD))
                                    # for OPTAA wavelengths, print the array
                                    if sv == 'wavelength_a' or sv == 'wavelength_c':
                                        [g_min, g_max] = cf.get_global_ranges(r, sv)
                                        n_all = len(var)
                                        mean = list(vD)
                                        num_outliers = None
                                        vmin = None
                                        vmax = None
                                        sd = None
                                        n_stats = 'not calculated'
                                        n_nan = None
                                        n_fv = None
                                        n_grange = 'no global ranges'
                                        fv = var['fv'][0]
                                    else:
                                        if vnum_dims > 2:
                                            print('variable has more than 2 dimensions')
                                            num_outliers = None
                                            mean = None
                                            vmin = None
                                            vmax = None
                                            sd = None
                                            n_stats = 'variable has more than 2 dimensions'
                                            n_nan = None
                                            n_fv = None
                                            n_grange = None
                                            fv = None
                                            n_all = None
                                        else:
                                            if vnum_dims > 1:
                                                n_all = [len(vD), len(vD.flatten())]
                                            else:
                                                n_all = len(vD)
                                            n_nan = int(np.sum(np.isnan(vD)))
                                            fv = var['fv'][0]
                                            vD[vD == fv] = np.nan  # turn fill values to nans
                                            n_fv = int(np.sum(np.isnan(vD))) - n_nan

                                            [g_min, g_max] = cf.get_global_ranges(r, sv)
                                            if list(np.unique(np.isnan(vD))) != [True]:
                                                # reject data outside of global ranges
                                                if g_min is not None and g_max is not None:
                                                    # turn data outside of global ranges to nans
                                                    #var_gr = var_nofv.where((var_nofv >= g_min) & (var_nofv <= g_max))
                                                    vD[vD < g_min] = np.nan
                                                    vD[vD > g_max] = np.nan
                                                    n_grange = int(np.sum(np.isnan(vD)) - n_fv - n_nan)
                                                else:
                                                    n_grange = 'no global ranges'

                                                if list(np.unique(np.isnan(vD))) != [True]:
                                                    if sv == 'spkir_abj_cspp_downwelling_vector':
                                                        # don't remove outliers from dataset
                                                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics_spkir(vD)
                                                    else:
                                                        if vnum_dims > 1:
                                                            var_gr = vD.flatten()
                                                        # drop nans before calculating stats
                                                        var_gr = var_gr[~np.isnan(var_gr)]
                                                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(var_gr, 5)
                                                else:
                                                    num_outliers = None
                                                    mean = None
                                                    vmin = None
                                                    vmax = None
                                                    sd = None
                                                    n_stats = 0
                                                    n_grange = None
                                            else:
                                                num_outliers = None
                                                mean = None
                                                vmin = None
                                                vmax = None
                                                sd = None
                                                n_stats = 0
                                                n_grange = None

                                    if vnum_dims > 1:
                                        sv = '{} (dims: {})'.format(sv, list(np.shape(var['values'])))
                                    else:
                                        sv = sv
                                    #if 'timedelta' not in str(var.values.dtype):
                                    data['deployments'][deployment]['method'][method]['stream'][data_stream]['file'][
                                        fname]['sci_var_stats'][sv] = dict(n_outliers=num_outliers, mean=mean, min=vmin,
                                                                           max=vmax, stdev=sd, n_stats=n_stats, units=var_units,
                                                                           n_nans=n_nan, n_fillvalues=n_fv, fill_value=str(fv),
                                                                           global_ranges=[g_min, g_max], n_grange=n_grange,
                                                                           n_all=n_all)

                    sfile = os.path.join(save_dir, '{}-{}-file_analysis.json'.format(rev_dep, r))
                    with open(sfile, 'w') as outfile:
                        json.dump(data, outfile)
                    json_file_list.append(str(sfile))

        depfile = os.path.join(save_dir, '{}-dependencies.txt'.format(r))
        with open(depfile, 'w') as depf:
            depf.write(str(dependencies))

    return json_file_list


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/output'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190904T191438036Z-CE02SHBP-LJ01D-08-OPTAAD106-streamed-optaa_sample/catalog.html']
    deployment_num = 2  # None or int

    main(sDir, url_list, deployment_num)
