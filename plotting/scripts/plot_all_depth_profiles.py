#!/usr/bin/env python
"""
Created on Feb 2019

@author: Leila Belabbassi
@brief: This script is used to create depth-profile plots for instruments data on mobile platforms (WFP & Gliders).
Each plot contains data from all deployments.
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
import matplotlib.cm as cm


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def main(url_list, sDir, plot_type):
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

        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # read in the analysis file
        dr_data = cf.refdes_datareview_json(r)

        # get end times of deployments
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        # get the list of data files and filter out collocated instruments and other streams chat
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
        separate the data files by methods
        '''
        for ms in ms_list:  # np.unique(methodstream)
            fdatasets_sel = [x for x in fdatasets if ms in x]

            # create a folder to save figures
            save_dir = os.path.join(sDir, array, subsite, r, plot_type, ms.split('-')[0])
            cf.create_dir(save_dir)

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

            y_unit = []
            y_name = []
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
                        t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
                        t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')

                        # sci variable
                        z = ds[var].values
                        sh['t'] = np.append(sh['t'], t)
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
                            if ds[pressure].units not in y_unit:
                                y_unit.append(ds[pressure].units)
                            if ds[pressure].long_name not in y_name:
                                y_name.append(ds[pressure].long_name)

                        sh['pressure'] = np.append(sh['pressure'], y)

            if len(y_unit) != 1:
                print('pressure unit varies!')
            else:
                y_unit = y_unit[0]

            if len(y_name) != 1:
                print('pressure long name varies!')
            else:
                y_name = y_name[0]

            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print('\nWorking on variable: {}'.format(sv))
                    if len(vinfo['t']) < 1:
                        print('no variable data to plot')
                    else:
                        sv_units = vinfo['units'][0]
                        fv = vinfo['fv'][0]
                        t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        t = vinfo['t']
                        x = vinfo['values']
                        y = vinfo['pressure']

                    # Check if the array is all NaNs
                    if sum(np.isnan(x)) == len(x):
                        print('Array of all NaNs - skipping plot.')
                        continue

                    # Check if the array is all fill values
                    elif len(x[x != fv]) == 0:
                        print('Array of all fill values - skipping plot.')
                        continue

                    else:
                        # reject fill values
                        fv_ind = x != fv
                        y_nofv = y[fv_ind]
                        t_nofv = t[fv_ind]
                        c_nofv = cm.rainbow(np.linspace(0, 1, len(t[fv_ind])))
                        x_nofv = x[fv_ind]
                        print(len(x) - len(fv_ind), ' fill values')

                        # reject NaNs
                        nan_ind = ~np.isnan(x)
                        t_nofv_nonan = t_nofv[nan_ind]
                        c_nofv_nonan = c_nofv[nan_ind]
                        y_nofv_nonan = y_nofv[nan_ind]
                        x_nofv_nonan = x_nofv[nan_ind]
                        print(len(x) - len(nan_ind), ' NaNs')

                        # reject extreme values
                        ev_ind = cf.reject_extreme_values(x_nofv_nonan)
                        t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
                        c_nofv_nonan_noev = c_nofv_nonan[ev_ind]
                        y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
                        x_nofv_nonan_noev = x_nofv_nonan[ev_ind]
                        print(len(z) - len(ev_ind), ' Extreme Values', '|1e7|')

                        # reject values outside global ranges:
                        global_min, global_max = cf.get_global_ranges(r, sv)
                        # platform not in qc-table (parad_k_par)
                        # global_min = 0
                        # global_max = 2500
                        print('global ranges for : {}-{}  {} - {}'.format(r, sv, global_min, global_max))
                        if isinstance(global_min, (int, float)) and isinstance(global_max, (int, float)):
                            gr_ind = cf.reject_global_ranges(x_nofv_nonan_noev, global_min, global_max)
                            t_nofv_nonan_noev_nogr = t_nofv_nonan_noev[gr_ind]
                            y_nofv_nonan_noev_nogr = y_nofv_nonan_noev[gr_ind]
                            x_nofv_nonan_noev_nogr = x_nofv_nonan_noev[gr_ind]
                        else:
                            t_nofv_nonan_noev_nogr = t_nofv_nonan_noev
                            y_nofv_nonan_noev_nogr = y_nofv_nonan_noev
                            x_nofv_nonan_noev_nogr = x_nofv_nonan_noev


                    if len(x_nofv_nonan_noev) > 0:
                        if m == 'common_stream_placeholder':
                            sname = '-'.join((r, sv))
                        else:
                            sname = '-'.join((r, m, sv))

                    if sv != 'pressure':
                        columns = ['tsec', 'dbar', str(sv)]
                        bin_size = 10
                        min_r = int(round(min(y_nofv_nonan_noev)-bin_size))
                        max_r = int(round(max(y_nofv_nonan_noev)+bin_size))
                        ranges = list(range(min_r, max_r, bin_size))
                        groups, d_groups = gt.group_by_depth_range(t_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr,
                                                                   x_nofv_nonan_noev_nogr, columns, ranges)

                    y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr = [], [], [], [], [], [], []
                    tm = 1
                    for ii in range(len(groups)):
                        nan_ind = d_groups[ii + tm].notnull()
                        xtime = d_groups[ii + tm][nan_ind]
                        colors = cm.rainbow(np.linspace(0, 1, len(xtime)))
                        ypres = d_groups[ii + tm + 1][nan_ind]
                        nval = d_groups[ii + tm + 2][nan_ind]
                        tm += 2

                        l_arr.append(len(nval))  # count of data to filter out small groups
                        y_avg.append(ypres.mean())
                        n_avg.append(nval.mean())
                        n_min.append(nval.min())
                        n_max.append(nval.max())
                        n_std = 3
                        n0_std.append(nval.mean() + n_std * nval.std())
                        n1_std.append(nval.mean() - n_std * nval.std())

                    # Plot all data
                    ylabel = y_name + " (" + y_unit + ")"
                    xlabel = sv + " (" + sv_units + ")"
                    clabel = 'Time'

                    fig, ax = pf.plot_profiles(x_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr, t_nofv_nonan_noev_nogr,
                                               ylabel, xlabel, clabel, end_times, deployments, stdev=None)

                    title_text = ' '.join((r, ms.split('-')[-1])) + '\n' \
                                 + t0 + ' - ' + t1 + '\n' + str(bin_size) +\
                                 ' m average and ' + str(n_std) + ' std shown'

                    ax.set_title(title_text, fontsize=9)
                    ax.plot(n_avg, y_avg, '-k')

                    ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                    pf.save_fig(save_dir, sname)

                    # Plot data with outliers removed

                    fig, ax = pf.plot_profiles(x_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr, t_nofv_nonan_noev_nogr,
                                               ylabel, xlabel, clabel, end_times, deployments, stdev=5)
                    ax.set_title(' '.join((r, ms.split('-')[-1])) + '\n' \
                                 + t0 + ' - ' + t1, fontsize=9)
                    sfile = '_'.join((sname,'rmoutliers'))
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T135948-CP02PMCO-WFP01-03-CTDPFK000-recovered_wfp-ctdpf_ckl_wfp_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T140002-CP02PMCO-WFP01-03-CTDPFK000-telemetered-ctdpf_ckl_wfp_instrument/catalog.html']

    plot_type = 'profile_plots'

    main(url_list, sDir, plot_type)