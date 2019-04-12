#!/usr/bin/env python
"""
Created on Feb 2019

@author: Leila Belabbassi
@brief: This script is used to create depth-profile plots for instruments data on mobile platforms (WFP & Gliders).
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
import matplotlib.pyplot as plt
import functions.group_by_timerange as gt


def main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num, zdbar, n_std, inpercentile, zcell_size):

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

        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # get end times of deployments
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

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

            texclude_dir = os.path.join(sDir, array, subsite, refdes, 'time_to_exclude')
            cf.create_dir(texclude_dir)

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

                    sh['t'] = np.append(sh['t'], ds['time'].values) # t = ds['time'].values
                    sh['values'] = np.append(sh['values'], ds[var].values)  # z = ds[var].values

                    y, y_unit, y_name = cf.add_pressure_to_dictionary_of_sci_vars(ds)
                    # if 'MOAS' in subsite:
                    #     if 'CTD' in main_sensor:  # for glider CTDs, pressure is a coordinate
                    #         pressure = 'sci_water_pressure_dbar'
                    #         y = ds[pressure].values
                    #     else:
                    #         pressure = 'int_ctd_pressure'
                    #         y = ds[pressure].values
                    # else:
                    #     pressure = pf.pressure_var(ds, ds.data_vars.keys())
                    #     y = ds[pressure].values
                    #
                    # if len(y[y != 0]) == 0 or sum(np.isnan(y)) == len(y) or len(y[y != ds[pressure]._FillValue]) == 0:
                    #     print('Pressure Array of all zeros or NaNs or fill values - using pressure coordinate')
                    #     pressure = [pressure for pressure in ds.coords.keys() if 'pressure' in ds.coords[pressure].name]
                    #     y = ds.coords[pressure[0]].values

                    sh['pressure'] = np.append(sh['pressure'], y)

                    # try:
                    #     ds[pressure].units
                    #     if ds[pressure].units not in y_unit:
                    #         y_unit.append(ds[pressure].units)
                    # except AttributeError:
                    #     print('pressure attributes missing units')
                    #     if 'pressure unit missing' not in y_unit:
                    #         y_unit.append('pressure unit missing')
                    #
                    # try:
                    #     ds[pressure].long_name
                    #     if ds[pressure].long_name not in y_name:
                    #         y_name.append(ds[pressure].long_name)
                    # except AttributeError:
                    #     print('pressure attributes missing long_name')
                    #     if 'pressure long name missing' not in y_name:
                    #         y_name.append('pressure long name missing')

            stat_data = pd.DataFrame(columns=['deployments', 'time_to_exclude'])
            file_exclude = '{}/{}_{}_{}_excluded_timestamps.csv'.format(texclude_dir,
                                                                                   deployment, refdes, method)
            stat_data.to_csv(file_exclude, index=True)
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
                        colors = cm.rainbow(np.linspace(0, 1, len(vinfo['t'])))
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
                        t__nofv = t[fv_ind]
                        y_nofv = y[fv_ind]
                        c_nofv = colors[fv_ind]
                        z_nofv = z[fv_ind]
                        print(len(z) - len(fv_ind), ' fill values')

                        # reject NaNs
                        nan_ind = ~np.isnan(z_nofv)
                        t_nofv_nonan = t__nofv[nan_ind]
                        c_nofv_nonan = c_nofv[nan_ind]
                        y_nofv_nonan = y_nofv[nan_ind]
                        z_nofv_nonan = z_nofv[nan_ind]
                        print(len(z) - len(nan_ind), ' NaNs')

                        # reject extreme values
                        ev_ind = cf.reject_extreme_values(z_nofv_nonan)
                        t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
                        c_nofv_nonan_noev = c_nofv_nonan[ev_ind]
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
                                  ' Global ranges: [{} - {}]'.format(global_min, global_max))
                        else:
                            gr_ind = []
                            t_nofv_nonan_noev_nogr = t_nofv_nonan_noev
                            y_nofv_nonan_noev_nogr = y_nofv_nonan_noev
                            z_nofv_nonan_noev_nogr = z_nofv_nonan_noev
                            print('No global ranges: [{} - {}]'.format(global_min, global_max))

                        # reject suspect data
                        columns = ['tsec', 'dbar', str(sv)]
                        min_r = int(round(min(y_nofv_nonan_noev_nogr) - zcell_size))
                        max_r = int(round(max(y_nofv_nonan_noev_nogr) + zcell_size))
                        ranges = list(range(min_r, max_r, zcell_size))
                        #     ... grouping by depth
                        groups, d_groups = gt.group_by_depth_range(t_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr,
                                                                   z_nofv_nonan_noev_nogr, columns, ranges)
                        #     ... excluding timestamps
                        if 'scatter' in sv:
                            n_std = None #to use percentile
                        else:
                            n_std = n_std

                        y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex, \
                        t_nofv_nonan_noev_nogr_nospct, z_nofv_nonan_noev_nogr_nospct, y_nofv_nonan_noev_nogr_nospct = \
                            cf.time_exclude_std(
                                                groups, d_groups, n_std,
                                                t_nofv_nonan_noev_nogr,
                                                y_nofv_nonan_noev_nogr,
                                                z_nofv_nonan_noev_nogr,
                                                inpercentile)
                        print('{} using {} percentile of data grouped in {} dbar segments'.format(
                            len(z_nofv_nonan_noev_nogr) - len(z_nofv_nonan_noev_nogr_nospct), inpercentile, zcell_size))

                        # writing timestamps to .csv file to use with data_range.py script
                        if len(time_ex) != 0:
                            t_exclude = time_ex[0]
                            for i in range(len(time_ex))[1:len(time_ex)]:
                                t_exclude = '{}, {}'.format(t_exclude, time_ex[i])

                            stat_data = pd.DataFrame({'deployments': deployment,
                                                      'time_to_exclude': t_exclude}, index=[sv])
                            stat_data.to_csv(file_exclude, index=True, mode='a', header=False)

                        # reject time ranges from data portal file export
                        t_nofv_nonan_noev_nogr_nospct_nomore, \
                        z_nofv_nonan_noev_nogr_nospct_nomore, \
                        y_nofv_nonan_noev_nogr_nospct_nomore = \
                            cf.time_exclude_portal(subsite, r, t_nofv_nonan_noev_nogr_nospct,
                                                        y_nofv_nonan_noev_nogr_nospct, z_nofv_nonan_noev_nogr_nospct)

                        print('{} using visual inspection of data'.format(
                        len(t_nofv_nonan_noev_nogr_nospct) - len(t_nofv_nonan_noev_nogr_nospct_nomore), inpercentile,
                        zcell_size))

                    # reject data in a depth range
                    if zdbar is not None:
                        y_ind = y_nofv_nonan_noev_nogr_nospct_nomore < zdbar
                        t_noy = t_nofv_nonan_noev_nogr_nospct_nomore[y_ind]
                        y_noy = y_nofv_nonan_noev_nogr_nospct_nomore[y_ind]
                        z_noy = z_nofv_nonan_noev_nogr_nospct_nomore[y_ind]

                    print('{} using visual inspection of data'.format(
                        len(t_nofv_nonan_noev_nogr_nospct) - len(t_nofv_nonan_noev_nogr_nospct_nomore),
                        inpercentile,
                        zcell_size))
                    # plot data
                    if len(z_nofv_nonan_noev_nogr_nospct) > 0:
                        if m == 'common_stream_placeholder':
                            sname = '-'.join((sv, r))
                        else:
                            sname = '-'.join((sv, r, m))

                    xlabel = sv + " (" + sv_units + ")"
                    ylabel = y_name[0] + " (" + y_unit[0] + ")"
                    clabel = 'Time'
                    title = ' '.join((deployment, r, m))

                    # Plot error free  data
                    fig, ax = pf.plot_profiles(z_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr, t_nofv_nonan_noev_nogr,
                                               ylabel, xlabel, clabel, end_times, deployments, stdev=None)
                    ax.set_title(title, fontsize=9)
                    ax.plot(n_avg, y_avg, '-k')
                    ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                    leg_text = (
                        'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}]'.format(
                                                                len(z) - len(fv_ind),
                                                                len(z) - len(nan_ind),
                                                                len(z) - len(ev_ind),
                                                                len(gr_ind),
                                                            global_min, global_max) + '\n' +
                        ('(black) data average in {} dbar segments'.format(zcell_size)) + '\n' +
                        ('(magenta) upper and lower {} percentile envelope in {} dbar segments'.format(inpercentile, zcell_size)),)
                    ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                    fig.tight_layout()
                    sfile = '_'.join(('rm_erroneous_data', sname))
                    pf.save_fig(save_dir, sfile)

                    # plot excluding time ranges for suspect data
                    if len(t_nofv_nonan_noev_nogr_nospct) != len(t_nofv_nonan_noev_nogr):
                        fig, ax = pf.plot_profiles(z_nofv_nonan_noev_nogr_nospct, y_nofv_nonan_noev_nogr_nospct,
                                                   t_nofv_nonan_noev_nogr_nospct,
                                                   ylabel, xlabel, clabel, end_times, deployments, stdev=None)

                        ax.set_title(title, fontsize=9)
                        leg_text = (
                        'removed {} in the upper and lower {} percentile of data grouped in {} dbar segments'.format(
                        len(z_nofv_nonan_noev_nogr) - len(z_nofv_nonan_noev_nogr_nospct), inpercentile, zcell_size),)
                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                        fig.tight_layout()
                        sfile = '_'.join(('rm_suspect_data', sname))
                        pf.save_fig(save_dir, sfile)

                    # plot excluding time ranges from data portal for suspect data
                    if len(t_nofv_nonan_noev_nogr_nospct_nomore) != len(t_nofv_nonan_noev_nogr_nospct):
                        fig, ax = pf.plot_profiles(z_nofv_nonan_noev_nogr_nospct_nomore,
                                                   y_nofv_nonan_noev_nogr_nospct_nomore,
                                                   t_nofv_nonan_noev_nogr_nospct_nomore,
                                                   ylabel, xlabel, clabel, end_times, deployments, stdev=None)
                        ax.set_title(title, fontsize=9)
                        leg_text = ('excluded suspect data using visual inspection of the data',)
                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                        fig.tight_layout()
                        sfile = '_'.join(('rm_v_suspect_data', sname))
                        pf.save_fig(save_dir, sfile)


                    # Plot excluding a selected depth value
                    if len(t) != len(t):
                        fig, ax = pf.plot_profiles(z_noy, y_noy, t_noy,
                                                   ylabel, xlabel, clabel, end_times, deployments, stdev=None)

                        ax.set_title(title, fontsize=9)
                        leg_text = ('removed data in range {} dbar'.format(zcell_size),)
                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                        fig.tight_layout()
                        sfile = '_'.join(('rm_depth_range', sname))
                        pf.save_fig(save_dir, sfile)

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

    '''
    define time range: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    '''
    start_time = None  # dt.datetime(2014, 12, 1)
    end_time = None  # dt.datetime(2015, 5, 2)

    '''
    define filters standard deviation, percentile, depth range
    '''
    n_std = None
    inpercentile = 5
    zdbar = None

    '''
    define the depth cell_size for data grouping 
    '''
    zcell_size = 10

    ''''
    define deployment number and method
    '''
    method_num = 'telemetered'  # 'recovered_wfp'
    deployment_num = 6

    '''
    define plot type, save-director y name and URL where data files live 
    '''
    plot_type = 'profile_plots'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    # url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html']
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021350-CE09OSPM-WFP01-04-FLORTK000-telemetered-flort_sample/catalog.html']

    '''
        call in main function with the above attributes
    '''
    main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num, zdbar, n_std, inpercentile, zcell_size)