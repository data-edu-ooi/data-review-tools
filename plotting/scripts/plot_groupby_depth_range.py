#!/usr/bin/env python
"""
Created on Feb 2019

@author: Leila Belabbassi
@brief: This script is used to create 3-D color scatter plots for instruments data on mobile platforms (WFP & Gliders).
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
from matplotlib import pyplot


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def main(url_list, sDir, plot_type, deployment_num, start_time, end_time):
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
        for ms in ms_list:
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
            y_unit = []
            y_name = []
            for fd in fdatasets_sel:
                ds = xr.open_dataset(fd, mask_and_scale=False)
                print(fd)

                if start_time is not None and end_time is not None:
                    ds = ds.sel(time=slice(start_time, end_time))
                    if len(ds['time'].values) == 0:
                        print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                        continue

                fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)

                if deployment_num is not None:
                    if int(deployment.split('0')[-1]) is not deployment_num:
                        print(type(int(deployment.split('0')[-1])), type(deployment_num))
                        continue

                save_dir = os.path.join(sDir, array, subsite, refdes, plot_type, ms.split('-')[0], deployment)
                cf.create_dir(save_dir)

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
                    print('pressure unit varies UHHHHHHHHH')
                else:
                    y_unit = y_unit[0]

                if len(y_name) != 1:
                    print('pressure long name varies UHHHHHHHHH')
                else:
                    y_name = y_name[0]

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

                            title = ' '.join((r, ms.split('-')[1]))

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
                            nan_ind = ~np.isnan(z)
                            t_nofv_nonan = t_nofv[nan_ind]
                            y_nofv_nonan = y_nofv[nan_ind]
                            z_nofv_nonan = z_nofv[nan_ind]
                            print(len(z) - len(nan_ind), ' NaNs')

                            # reject extreme values
                            ev_ind = cf.reject_extreme_values(z_nofv_nonan)
                            t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
                            colors = cm.rainbow(np.linspace(0, 1, len(t_nofv_nonan_noev)))
                            y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
                            z_nofv_nonan_noev = z_nofv_nonan[ev_ind]
                            print(len(z) - len(ev_ind), ' Extreme Values', '|1e7|')

                        if len(y_nofv_nonan_noev) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((r, sv))
                            else:
                                sname = '-'.join((r, m, sv))
                        # Plot all data
                        ylabel = y_name + " (" + y_unit + ")"
                        xlabel = sv + " (" + sv_units + ")"
                        clabel = 'Time'
                        clabel = sv + " (" + sv_units + ")"

                        fig, ax = pf.plot_profiles(z_nofv_nonan_noev, y_nofv_nonan_noev, colors, xlabel, ylabel, stdev=None)
                        ax.set_title((title + '\n' + str(deployment_num) + ': ' +t0 + ' - ' + t1 + '\n' +
                                      'used bin = 2 dbar to calculate an average profile (black line) and 3-STD envelope (shaded area)'), fontsize=9)

                        # group by depth range
                        columns = ['time', 'pressure', str(sv)]
                        # ranges = [0, 50, 100, 200, 400, 600]
                        ranges = list(range(int(round(min(y_nofv_nonan_noev))), int(round(max(y_nofv_nonan_noev))), 1))
                        groups, d_groups = gt.group_by_depth_range(t_nofv_nonan_noev, y_nofv_nonan_noev, z_nofv_nonan_noev, columns, ranges)

                        # describe_file = '_'.join((sname, 'statistics.csv'))
                        # # groups.describe().to_csv(save_dir + '/' + describe_file)
                        ind = groups.describe()[sv]['mean'].notnull()
                        groups.describe()[sv][ind].to_csv('{}/{}_statistics.csv'.format(save_dir, sname), index=True)

                        tm = 1
                        fig, ax = pyplot.subplots(nrows=2, ncols=1)
                        pyplot.margins(y=.08, x=.02)
                        pyplot.grid()
                        y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr = [], [], [], [], [], [], []

                        for ii in range(len(groups)):

                            nan_ind = d_groups[ii+tm].notnull()
                            xtime = d_groups[ii+tm][nan_ind]
                            colors = cm.rainbow(np.linspace(0, 1, len(xtime)))
                            ypres = d_groups[ii+tm+1][nan_ind]
                            nval = d_groups[ii+tm+2][nan_ind]
                            tm += 2

                            # fig, ax = pf.plot_xsection(subsite, xtime, ypres, nval, clabel, ylabel, stdev=None)
                            # ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)

                            # pf.plot_profiles(nval, ypres, colors, ylabel, clabel, stdev=None)
                            # ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)

                            ind2 = cf.reject_outliers(nval, 5)
                            xD = nval[ind2]
                            yD = ypres[ind2]
                            nZ = colors[ind2]
                            outliers = str(len(nval) - len(xD))
                            leg_text = ('removed {} outliers (SD={})'.format(outliers, stdev),)

                            ax.scatter(xD, yD, c=nZ, s=2, edgecolor='None')
                            ax.invert_yaxis()
                            ax.set_xlabel(clabel, fontsize=9)
                            ax.set_ylabel(ylabel, fontsize=9)
                            ax.legend(leg_text, loc='best', fontsize=6)
                            ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)

                            l_arr.append(len(nval)) #  count of data to filter out small groups
                            y_avg.append(ypres.mean())
                            n_avg.append(nval.mean())
                            n_min.append(nval.min())
                            n_max.append(nval.max())
                            n0_std.append(nval.mean() + 3 * nval.std())
                            n1_std.append(nval.mean() - 3 * nval.std())


                        ax.plot(n_avg, y_avg, '-k')
                        # ax.plot(n_min, y_avg, '-b')
                        # ax.plot(n_max, y_avg, '-b')
                        ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                        sfile = '_'.join((sname, 'statistics'))
                        pf.save_fig(save_dir, sfile)

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T135158-CP01CNPM-WFP01-03-CTDPFK000-telemetered-ctdpf_ckl_wfp_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T135146-CP01CNPM-WFP01-03-CTDPFK000-recovered_wfp-ctdpf_ckl_wfp_instrument_recovered/catalog.html']

    # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235146-CP03ISSM-MFD37-03-CTDBPD000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html']
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T161432-CE09OSPM-WFP01-03-CTDPFK000-recovered_wfp-ctdpf_ckl_wfp_instrument_recovered/catalog.html']
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T161444-CE09OSPM-WFP01-03-CTDPFK000-telemetered-ctdpf_ckl_wfp_instrument/catalog.html'
    plot_type = 'profile_statistic_plots'

    start_time = None  # dt.datetime(2016, 6, 1, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None       # dt.datetime(2017, 10, 1, 0, 0, 0)  # optional, set to None if plotting all data
    deployment_num = None # 1

    main(url_list, sDir, plot_type, deployment_num, start_time, end_time)
