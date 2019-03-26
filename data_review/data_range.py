#!/usr/bin/env python
"""
Created on March 2019

@author: Leila Belabbassi
@brief: This script is used to generate a csv file with data ranges for instruments data on mobile platforms (WFP & Gliders).
Data ranges are calculated for user selected depth-bin size (e.g., bin = 10 dbar).
"""

import functions.plotting as pf
import functions.common as cf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt
import os
import pandas as pd
import itertools
import numpy as np
import xarray as xr
import datetime


def main(url_list, mDir, bin_size, zdbar):
    """""
    URL : path to instrument data by methods
    sDir : path to the directory on your machine to save files
    plot_type: folder name for a plot type

    """""
    rd_list = []
    ms_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = uu.split(rd + '-')[1].split('/')[0]
        if rd not in rd_list:
            rd_list.append(rd)
        if ms not in ms_list:
            ms_list.append(ms)

    ''' 
    separate different instruments
    '''
    for r in rd_list:
        print('\n{}'.format(r))
        subsite = r.split('-')[0]
        array = subsite[0:2]
        main_sensor = r.split('-')[-1]

        # read in the analysis file
        dr_data = cf.refdes_datareview_json(r)

        # get preferred stream
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # get end times of deployments
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))


        # get the list of data files and filter out collocated instruments and other streams
        datasets = []
        for u in url_list:
            print(u)
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)

        datasets = list(itertools.chain(*datasets))
        fdatasets = cf.filter_collocated_instruments(main_sensor, datasets)
        fdatasets = cf.filter_other_streams(r, ms_list, fdatasets)

        '''
        separate data files by methods
        '''
        for ms in ms_list:

            # create data ranges foe preferred data streams
            if (ms.split('-')[0]) == (ps_df[0].values[0].split('-')[0]):
                fdatasets_sel = [x for x in fdatasets if ms in x]

                # create a dictionary for science variables from analysis file
                stream_sci_vars_dict = dict()
                for x in dr_data['instrument']['data_streams']:
                    dr_ms = '-'.join((x['method'], x['stream_name']))
                    if ms == dr_ms:
                        stream_sci_vars_dict[dr_ms] = dict(vars=dict())
                        sci_vars = dict()
                        for y in x['stream']['parameters']:
                            if y['data_product_type'] == 'Science Data':
                                sci_vars.update({y['name']: dict(db_units=y['unit'])})
                        if len(sci_vars) > 0:
                            stream_sci_vars_dict[dr_ms]['vars'] = sci_vars

                # initialize an empty data array for science variables in dictionary
                sci_vars_dict = cd.initialize_empty_arrays(stream_sci_vars_dict, ms)

                print('\nAppending data from files: {}'.format(ms))

                for fd in fdatasets_sel:
                    ds = xr.open_dataset(fd, mask_and_scale=False)
                    print('\nAppending data file: {}'.format(fd.split('/')[-1]))
                    for var in list(sci_vars_dict[ms]['vars'].keys()):
                        sh = sci_vars_dict[ms]['vars'][var]
                        if ds[var].units == sh['db_units']:
                            if ds[var]._FillValue not in sh['fv']:
                                sh['fv'].append(ds[var]._FillValue)
                            if ds[var].units not in sh['units']:
                                sh['units'].append(ds[var].units)

                            # time
                            t = ds['time'].values
                            sh['t'] = np.append(sh['t'], t)

                            # sci variable
                            z = ds[var].values
                            sh['values'] = np.append(sh['values'], z)

                            # add pressure to dictionary of sci vars
                            if 'MOAS' in subsite:
                                if 'CTD' in main_sensor:  # for glider CTDs, pressure is a coordinate
                                    pressure = 'sci_water_pressure_dbar'
                                    y = ds[pressure].values
                                    if ds[pressure].units not in y_unit:
                                        y_unit.append(ds[pressure].units)
                                    if ds[pressure].long_name not in y_name:
                                        y_name.append(ds[pressure].long_name)
                                else:
                                    pressure = 'int_ctd_pressure'
                                    y = ds[pressure].values
                                    if ds[pressure].units not in y_unit:
                                        y_unit.append(ds[pressure].units)
                                    if ds[pressure].long_name not in y_name:
                                        y_name.append(ds[pressure].long_name)
                            else:
                                pressure = pf.pressure_var(ds, ds.data_vars.keys())
                                y = ds[pressure].values

                            if len(y[y != 0]) == 0 or sum(np.isnan(y)) == len(y) or len(y[y != ds[pressure]._FillValue]) == 0:
                                print('Pressure Array of all zeros or NaNs or fill values - using pressure coordinate')
                                pressure = [pressure for pressure in ds.coords.keys() if
                                            'pressure' in ds.coords[pressure].name]
                                y = ds.coords[pressure[0]].values

                            sh['pressure'] = np.append(sh['pressure'], y)


                # analyse data
                # create a folder to save data ranges
                save_dir_stat = os.path.join(mDir, array, subsite)
                cf.create_dir(save_dir_stat)
                stat_df = pd.DataFrame()

                for m, n in sci_vars_dict.items():
                    for sv, vinfo in n['vars'].items():
                        print(sv)
                        if len(vinfo['t']) < 1:
                            print('no variable data to plot')
                        else:
                            fv = vinfo['fv'][0]
                            t = vinfo['t']
                            z = vinfo['values']
                            y = vinfo['pressure']

                        # Check if the array is all NaNs
                        if sum(np.isnan(z)) == len(z):
                            print('Array of all NaNs - skipping plot.')
                            continue
                        # Check if the array is all fill values
                        elif len(z[z != fv]) == 0:
                            print('Array of all fill values - skipping plot.')
                            continue
                        else:
                            # reject fill values
                            fv_ind = z != fv
                            y_nofv = y[fv_ind]
                            t_nofv = t[fv_ind]
                            z_nofv = z[fv_ind]
                            print(len(z) - len(fv_ind), ' fill values')

                            # reject NaNs
                            nan_ind = ~np.isnan(z_nofv)
                            t_nofv_nonan = t_nofv[nan_ind]
                            y_nofv_nonan = y_nofv[nan_ind]
                            z_nofv_nonan = z_nofv[nan_ind]
                            print(len(z_nofv) - len(nan_ind), ' NaNs')

                            # reject extreme values
                            ev_ind = cf.reject_extreme_values(z_nofv_nonan)
                            t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
                            y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
                            z_nofv_nonan_noev = z_nofv_nonan[ev_ind]
                            print(len(z_nofv_nonan) - len(ev_ind), ' Extreme Values', '|1e7|')

                            # reject values outside global ranges:
                            global_min, global_max = cf.get_global_ranges(r, sv)
                            # platform not in qc-table (parad_k_par)
                            # global_min = 0
                            # global_max = 2500
                            if isinstance(global_min, (int, float)) and isinstance(global_max, (int, float)):
                                gr_ind = cf.reject_global_ranges(z_nofv_nonan_noev, global_min, global_max)
                                t_nofv_nonan_noev_nogr = t_nofv_nonan_noev[gr_ind]
                                y_nofv_nonan_noev_nogr = y_nofv_nonan_noev[gr_ind]
                                z_nofv_nonan_noev_nogr = z_nofv_nonan_noev[gr_ind]
                                print(len(z_nofv_nonan_noev) - len(gr_ind),
                                      ' Global ranges for : {} - {}'.format(global_min, global_max))
                            else:
                                t_nofv_nonan_noev_nogr = t_nofv_nonan_noev
                                y_nofv_nonan_noev_nogr = y_nofv_nonan_noev
                                z_nofv_nonan_noev_nogr = z_nofv_nonan_noev
                                print('No global ranges: {} - {}'.format(global_min, global_max))


                            #reject excluded time ranges
                            dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
                            drn = dr.loc[dr.type == 'exclusion']
                            if len(drn) != 0:
                                subsite_node = '-'.join((subsite, r.split('-')[1]))
                                drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]

                                t_ex = t_nofv_nonan_noev_nogr
                                y_ex = y_nofv_nonan_noev_nogr
                                z_ex = z_nofv_nonan_noev_nogr
                                for i, row in drne.iterrows():
                                    sdate = cf.format_dates(row.start_date)
                                    edate = cf.format_dates(row.end_date)
                                    ts = np.datetime64(sdate)
                                    te = np.datetime64(edate)
                                    ind = np.where((t_ex < ts) | (t_ex > te), True, False)
                                    if len(ind) != 0:
                                        t_ex = t_ex[ind]
                                        z_ex = z_ex[ind]
                                        y_ex = y_ex[ind]
                                        print(len(ind), 'timestamps in: {} - {}'.format(sdate, edate))
                            else:
                                print(len(z_ex), 'no time ranges excluded -  Empty Array', drn)

                            # Plot data for a selected depth range
                            if zdbar is not None:
                                y_ind = y_ex < zdbar
                                t_y = t_ex[y_ind]
                                y_y = y_ex[y_ind]
                                z_y = z_ex[y_ind]
                            else:
                                t_y = t_ex
                                y_y = y_ex
                                z_y = z_ex

                        # create data ranges for non - pressure data only
                        if 'pressure' not in sv:
                            columns = ['tsec', 'dbar', str(sv)]
                            # create depth ranges
                            min_r = int(round(min(y_y) - bin_size))
                            max_r = int(round(max(y_y) + bin_size))
                            ranges = list(range(min_r, max_r, bin_size))

                            # group data by depth
                            groups, d_groups = gt.group_by_depth_range(t_y, y_y,
                                                                       z_y, columns, ranges)

                            print('writing data ranges for {}'.format(sv))
                            stat_data = groups.describe()[sv]
                            stat_data.insert(loc=0, column='parameter', value=sv, allow_duplicates=False)
                            t_deploy = deployments[0]
                            for i in range(len(deployments))[1:len(deployments)]:
                                t_deploy = '{}, {}'.format(t_deploy, deployments[i])

                            stat_data.insert(loc=1, column='deployments', value=t_deploy, allow_duplicates=False)
                            stat_df = stat_df.append(stat_data)


                    # write stat file
                    stat_df.to_csv('{}/{}_data_ranges.csv'.format(save_dir_stat, r), index=True, float_format='%11.6f')


if __name__ == '__main__':
    bin_size = 10
    zdbar = None
    mDir = '/Users/leila/Documents/NSFEduSupport/github/data-review-tools/data_review/data_ranges'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T135500-CP01CNSP-SP001-06-DOSTAJ000-recovered_cspp-dosta_abcdjm_cspp_instrument_recovered/catalog.html']
    main(url_list, mDir, bin_size, zdbar)
