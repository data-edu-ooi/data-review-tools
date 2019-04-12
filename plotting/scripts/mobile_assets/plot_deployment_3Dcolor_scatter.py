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
            # y_unit = []
            # y_name = []

            for var in list(sci_vars_dict[ms]['vars'].keys()):
                sh = sci_vars_dict[ms]['vars'][var]
                if ds[var].units == sh['db_units']:
                    if ds[var]._FillValue not in sh['fv']:
                        sh['fv'].append(ds[var]._FillValue)
                    if ds[var].units not in sh['units']:
                        sh['units'].append(ds[var].units)

                    sh['t'] = np.append(sh['t'], ds['time'].values)
                    sh['values'] = np.append(sh['values'], ds[var].values)

                    y, y_unit, y_name = cf.add_pressure_to_dictionary_of_sci_vars(ds)


            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print('\n'+'working on ', sv)
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
                            # reject erroneous data
                            dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max = \
                                                                            cf.reject_erroneous_data(r, sv, t, y, z, fv)

                            # creating data groups
                            columns = ['tsec', 'dbar', str(sv)]
                            min_r = int(round(min(zpressure) - zcell_size))
                            max_r = int(round(max(zpressure) + zcell_size))
                            ranges = list(range(min_r, max_r, zcell_size))

                            groups, d_groups = gt.group_by_depth_range(dtime,  zpressure, ndata, columns, ranges)

                            #  rejecting timestamps from percentile analysis
                            y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex, \
                            t_nospct, z_nospct, y_nospct = cf.reject_timestamps_in_groups(groups, d_groups, n_std,
                                                                                           dtime, zpressure, ndata,
                                                                                           inpercentile)
                            print('{} using {} percentile of data grouped in {} dbar segments'.format(
                                                        len(zpressure) - len(z_nospct), inpercentile, zcell_size))

                            # reject time range from data portal file export
                            t_portal, z_portal, y_portal = cf.reject_timestamps_dataportal(subsite, r,
                                                                                        t_nospct, z_nospct, y_nospct)
                            print('{} using visual inspection of data'.format(len(z_nospct) - len(z_portal),
                                                                              inpercentile, zcell_size))

                        ''''
                        Plot data
                        '''''

                        if len(t_portal) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((sv, r))
                            else:
                                sname = '-'.join((sv, r, m))

                        clabel = sv + " (" + sv_units + ")"
                        ylabel = y_name[0] + " (" + y_unit[0] + ")"


                        # plot non-erroneous data
                        fig, ax, bar = pf.plot_xsection(subsite, dtime, zpressure, ndata,
                                                        clabel, ylabel, inpercentile, stdev=None)
                        ax.set_title(title, fontsize=9)
                        leg_text = (
                            'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}]'.format
                                (
                                len(z) - lenfv, len(z) - lennan, len(z) - lenev, lengr, global_min, global_max) + '\n' +
                                ('(black) data average in {} dbar segments'.format(zcell_size)) + '\n' +
                                ('(magenta) upper and lower {} percentile envelope in {} dbar segments'.format(
                                                                                            inpercentile, zcell_size)),
                                )
                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                        fig.tight_layout()
                        sfile = '_'.join(('rm_erroneous_data', sname))
                        pf.save_fig(save_dir, sfile)

                        # plot excluding timestamps for suspect data
                        if len(z_nospct) != len(zpressure):
                            fig, ax, bar = pf.plot_xsection(subsite, t_nospct, y_nospct, z_nospct,
                                                            clabel, ylabel, inpercentile=None, stdev=None)

                            ax.set_title(title, fontsize=9)
                            leg_text = (
                            'removed {} in the upper and lower {} percentile of data grouped in {} dbar segments'.format(
                            len(zpressure) - len(z_nospct), inpercentile, zcell_size),)
                            ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            fig.tight_layout()
                            sfile = '_'.join(('rm_suspect_data', sname))
                            pf.save_fig(save_dir, sfile)

                        # plot excluding time ranges from data portal export
                        if len(z_nospct) - len(z_portal) > 0:

                            fig, ax, bar = pf.plot_xsection(subsite, t_portal, y_portal, z_portal,
                                                            clabel, ylabel, inpercentile=None, stdev=None)
                            ax.set_title(title, fontsize=9)
                            leg_text = ('excluded {} suspect data when inspected visually'.format(len(z_nospct) - len(z_portal)),)
                            ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            fig.tight_layout()
                            sfile = '_'.join(('rm_v_suspect_data', sname))
                            pf.save_fig(save_dir, sfile)

                        # Plot data excluding a depth range
                        if zdbar is not None:
                            y_ind = y_portal < zdbar
                            t_zrange = t_portal[y_ind]
                            y_zrange = y_portal[y_ind]
                            z_zrange = z_portal[y_ind]

                            fig, ax, bar = pf.plot_xsection(subsite, t_zrange, y_zrange, z_zrange,
                                                            clabel, ylabel, inpercentile, stdev=None)
                            ax.set_title(title, fontsize=9)
                            leg_text = ('removed data in range {} dbar'.format(zcell_size),)
                            ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            fig.tight_layout()

                            sfile = '_'.join(('rm_depth_range', sname))
                            pf.save_fig(save_dir, sfile)

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

    """
    define time range: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    """
    start_time = None #dt.datetime(2014, 12, 1)
    end_time = None #dt.datetime(2015, 5, 2)

    '''
    define filters standard deviation, percentile, depth range
    '''

    zdbar = None
    n_std = None
    inpercentile = 5

    '''
    define the depth cell_size for data grouping 
    '''
    zcell_size = 10

    ''''
    define deployment number and collection method
    '''
    method_num = 'recovered_wfp'
    #method_num = 'telemetered'
    deployment_num = 1

    '''
    define plot type, output directory, and data files URL 
    '''
    plot_type = 'xsection_plots'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html']
    #url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021350-CE09OSPM-WFP01-04-FLORTK000-telemetered-flort_sample/catalog.html']

    '''
    call in main function with the above attributes
    '''
    main(url_list, sDir, plot_type, deployment_num, start_time, end_time, method_num, zdbar, n_std, inpercentile, zcell_size)