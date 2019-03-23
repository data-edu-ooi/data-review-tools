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
import matplotlib.cm as cm
import datetime as dt
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd


def main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num):

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
            print('\nDataset {} of {}: {}'.format(ii + 1, len(datasets_sel), d))
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

                    if len(y[y != 0]) == 0 or sum(np.isnan(y)) == len(y) or en(y[y != ds[pressure]._FillValue]) == 0:
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

                            if len(y_nofv_nonan_noev) > 0:
                                if m == 'common_stream_placeholder':
                                    sname = '-'.join((r, sv))
                                else:
                                    sname = '-'.join((r, m, sv))

                            # Plot all data
                            print(y_name)
                            clabel = sv + " (" + sv_units + ")"
                            ylabel = y_name[0] + " (" + y_unit[0] + ")"
                            print(clabel, ylabel)
                            fig, ax = pf.plot_xsection(subsite, t_nofv_nonan_noev, y_nofv_nonan_noev, z_nofv_nonan_noev,
                                                       clabel, ylabel, stdev=None)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            pf.save_fig(save_dir, sname)

                            # Plot data with outliers removed
                            fig, ax = pf.plot_xsection(subsite, t_nofv_nonan_noev, y_nofv_nonan_noev, z_nofv_nonan_noev,
                                                       clabel, ylabel, stdev=5)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                            sfile = '_'.join((sname, 'rmoutliers'))
                            pf.save_fig(save_dir, sfile)

                            # plot data with excluded time range removed
                            dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
                            drn = dr.loc[dr.type == 'exclusion']
                            if len(drn) != 0:
                                subsite_node = '-'.join((subsite, r.split('-')[1]))
                                drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]

                                t_ex = t_nofv_nonan_noev
                                y_ex = y_nofv_nonan_noev
                                z_ex = z_nofv_nonan_noev
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

                                fig, ax = pf.plot_profiles(z_ex, y_ex, t_ex,
                                                           ylabel, xlabel, clabel, end_times, deployments, stdev=None)
                                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)

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
    start_time = dt.datetime(2014, 12, 1)
    end_time = dt.datetime(2015, 5, 2)
    method_num = 'recovered_wfp'
    deployment_num = 2
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = []
    main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num)
