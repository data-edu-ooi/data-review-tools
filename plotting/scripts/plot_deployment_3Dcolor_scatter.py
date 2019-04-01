#!/usr/bin/env python
"""
Created on Feb 2019

@author: Leila Belabbassi
@brief: This script is used to create 3-D color scatter plots for instruments data on mobile platforms (WFP & Gliders).
Each plot contain data from one deployment.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt

def main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num, zdbar, n_std):

    for i, u in enumerate(url_list):
        print('\nUrl {} of {}: {}'.format(i + 1, len(url_list), u))
        elements = u.split('/')[-2].split('-')
        r = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = u.split(r + '-')[1].split('/')[0]
        subsite = r.split('-')[0]
        array = subsite[0:2]
        main_sensor = r.split('-')[-1]

        # read URL to get data
        datasets = cf.get_nc_urls([u])
        datasets_sel = cf.filter_collocated_instruments(main_sensor, datasets)

        # get sci data review list
        dr_data = cf.refdes_datareview_json(r)

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


        for ii, d in enumerate(datasets_sel):
            part_d = d.split('/')[-1]
            print('\nDataset {} of {}: {}'.format(ii + 1, len(datasets_sel), part_d))
            with xr.open_dataset(d, mask_and_scale=False) as ds:
                ds = ds.swap_dims({'obs': 'time'})

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)

            if method_num is not None:
                if method != method_num:
                    print(method_num, method)
                    continue

            if deployment_num is not None:
                if int(deployment.split('0')[-1]) is not deployment_num:
                    print(type(int(deployment.split('0')[-1])), type(deployment_num))
                    continue

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue
                stime = start_time.strftime('%Y-%m-%d')
                etime = end_time.strftime('%Y-%m-%d')
                ext = stime + 'to' + etime  # .join((ds0_method, ds1_method
                save_dir = os.path.join(sDir, array, subsite, refdes, plot_type, ms.split('-')[0], deployment, ext)
            else:
                save_dir = os.path.join(sDir, array, subsite, refdes, plot_type, ms.split('-')[0], deployment)

            cf.create_dir(save_dir)

            # initialize an empty data array for science variables in dictionary
            sci_vars_dict = cd.initialize_empty_arrays(stream_sci_vars_dict, ms)
            y_unit = []
            y_name = []

            for var in list(sci_vars_dict[ms]['vars'].keys()):
                sh = sci_vars_dict[ms]['vars'][var]
                if ds[var].units == sh['db_units']:
                    if ds[var]._FillValue not in sh['fv']:
                        sh['fv'].append(ds[var]._FillValue)
                    if ds[var].units not in sh['units']:
                        sh['units'].append(ds[var].units)

                    sh['t'] = np.append(sh['t'], ds['time'].values) #t = ds['time'].values
                    sh['values'] = np.append(sh['values'], ds[var].values)  # z = ds[var].values

                    if 'MOAS' in subsite:
                        if 'CTD' in main_sensor:  # for glider CTDs, pressure is a coordinate
                            pressure = 'sci_water_pressure_dbar'
                            y = ds[pressure].values
                        else:
                            pressure = 'int_ctd_pressure'
                            y = ds[pressure].values
                    else:
                        pressure = pf.pressure_var(ds, ds.data_vars.keys())
                        y = ds[pressure].values

                    if len(y[y != 0]) == 0 or sum(np.isnan(y)) == len(y) or len(y[y != ds[pressure]._FillValue]) == 0:
                        print('Pressure Array of all zeros or NaNs or fill values - using pressure coordinate')
                        pressure = [pressure for pressure in ds.coords.keys() if 'pressure' in ds.coords[pressure].name]
                        y = ds.coords[pressure[0]].values

                    sh['pressure'] = np.append(sh['pressure'], y)

                    try:
                        ds[pressure].units
                        if ds[pressure].units not in y_unit:
                            y_unit.append(ds[pressure].units)
                    except AttributeError:
                        print('pressure attributes missing units')
                        if 'pressure unit missing' not in y_unit:
                            y_unit.append('pressure unit missing')

                    try:
                        ds[pressure].long_name
                        if ds[pressure].long_name not in y_name:
                            y_name.append(ds[pressure].long_name)
                    except AttributeError:
                        print('pressure attributes missing long_name')
                        if 'pressure long name missing' not in y_name:
                            y_name.append('pressure long name missing')


            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print(sv)
                    if len(vinfo['t']) < 1:
                        print('no variable data to plot')
                    else:
                        sv_units = vinfo['units'][0]
                        fv = vinfo['fv'][0]
                        t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        t = vinfo['t']
                        z = vinfo['values']
                        y = vinfo['pressure']

                        title = ' '.join((deployment, r, ms.split('-')[0]))

                        # Check if the array is all NaNs
                        if sum(np.isnan(z)) == len(z):
                            print('Array of all NaNs - skipping plot.')

                        # Check if the array is all fill values
                        elif len(z[z != fv]) == 0:
                            print('Array of all fill values - skipping plot.')

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
                            print(len(z) - len(nan_ind), ' NaNs')

                            # reject extreme values
                            ev_ind = cf.reject_extreme_values(z_nofv_nonan)
                            t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
                            y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
                            z_nofv_nonan_noev = z_nofv_nonan[ev_ind]
                            print(len(z) - len(ev_ind), ' Extreme Values', '|1e7|')

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
                                      ' Global Ranges [{} - {}]'.format(global_min, global_max))
                            else:
                                t_nofv_nonan_noev_nogr = t_nofv_nonan_noev
                                y_nofv_nonan_noev_nogr = y_nofv_nonan_noev
                                z_nofv_nonan_noev_nogr = z_nofv_nonan_noev
                                print('No global ranges: {} - {}'.format(global_min, global_max))

                            # group data by depth
                            columns = ['tsec', 'dbar', str(sv)]
                            bin_size = 10
                            min_r = int(round(min(y_nofv_nonan_noev_nogr) - bin_size))
                            max_r = int(round(max(y_nofv_nonan_noev_nogr) + bin_size))
                            ranges = list(range(min_r, max_r, bin_size))
                            groups, d_groups = gt.group_by_depth_range(t_nofv_nonan_noev_nogr,
                                                                       y_nofv_nonan_noev_nogr,
                                                                       z_nofv_nonan_noev_nogr, columns, ranges)

                            # reject values outside 3 STD in data groups
                            y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, \
                                time_ex, t_std, z_std, y_std = cf.time_exclude_std(groups, d_groups, n_std,
                                                 t_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr, z_nofv_nonan_noev_nogr)

                        # Plot data
                        if len(y_nofv_nonan_noev) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((sv, r))
                            else:
                                sname = '-'.join((sv, r, m))


                        clabel = sv + " (" + sv_units + ")"
                        ylabel = y_name[0] + " (" + y_unit[0] + ")"


                        # plot non-erroneous data
                        fig, ax, bar = pf.plot_xsection(subsite, t_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr,
                                                        z_nofv_nonan_noev_nogr, clabel, ylabel, stdev=None)
                        ax.set_title((title +
                                      '\n' + 'excluded : fill values, nans, |1e7| values, non-global ranges values'),
                                     fontsize=9)

                        sfile = '_'.join(('rm_erroneous_data', sname))
                        pf.save_fig(save_dir, sfile)

                        # Plot data excluding a depth range
                        if zdbar is not None:
                            y_ind = y_nofv_nonan_noev_nogr < zdbar
                            t_y = t_nofv_nonan_noev_nogr[y_ind]
                            y_y = y_nofv_nonan_noev_nogr[y_ind]
                            z_y = z_nofv_nonan_noev_nogr[y_ind]

                            fig, ax, bar = pf.plot_xsection(subsite, t_y, y_y, z_y, clabel, ylabel, stdev=None)
                            ax.set_title((title +
                                          '\n' + 'excluded : fill values, nans, |1e7| values, non-global ranges values' +
                                          '\n' + 'removed data below ' + str(zdbar) + ' dbar'), fontsize=9)
                            pf.save_fig(save_dir, sfile)

                        # plot data excluding time ranges using the STD analysis
                        if len(t_std) != 0:
                            fig, ax, bar = pf.plot_xsection(subsite, t_std, y_std, z_std, clabel, ylabel, stdev=None)
                            ax.set_title((title + '\n' +
                                          'excluded suspect data using a ' +
                                          str(n_std) + ' STD filter on a bin-depth = ' + str(bin_size) + ' dbar'),
                                         fontsize=9)
                            leg_text = ('excluded suspect data',)
                            ax.legend(leg_text, loc='best', fontsize=6)
                            sfile = '_'.join(('rm_suspect_data', sname))
                            pf.save_fig(save_dir, sfile)

                        # plot data excluding time range using file export from the data review portal
                        t_ex, z_ex, y_ex = cf.time_exclude_portal(subsite, r, t_nofv_nonan_noev_nogr,
                                                                         y_nofv_nonan_noev_nogr, z_nofv_nonan_noev_nogr)
                        if t_ex is not None:
                            fig, ax = pf.plot_xsection(subsite, t_ex, y_ex, z_ex, clabel, ylabel, stdev=None)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            leg_text = ('excluded suspect data',)
                            ax.legend(leg_text, loc='best', fontsize=6)
                            sfile = '_'.join((sname, 'rmsuspectdata'))
                            pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    plot_type = 'xsection_plots'
    '''
    time option: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    '''
    start_time = None #dt.datetime(2014, 12, 1)
    end_time = None #dt.datetime(2015, 5, 2)
    n_std = 3
    method_num = 'recovered_wfp'
    deployment_num = 3
    zdbar = None

    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html']

    main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num, zdbar, n_std)