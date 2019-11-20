#!/usr/bin/env python
"""
Created on Feb 2019 by Leila Belabbassi
Modified on Nov 18 2019 by Lori Garzio

@brief: This script is used to create initial profile plots and 3D color scatter plots for instruments on cabled
profilers. Excludes erroneous data and data
outside of global ranges. Each plot contains data from one deployment and one science variable.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import itertools
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
import functions.group_by_timerange as gt
import functions.profile_xsection_spkir_optaa as pxso


def main(url_list, sDir, deployment_num, start_time, end_time, preferred_only, n_std, inpercentile, zcell_size, zdbar):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list and 'ENG' not in rd and 'ADCP' not in rd:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        deployments = []
        for url in url_list:
            splitter = url.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            catalog_rms = '-'.join((r, splitter[-2], splitter[-1]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([url])
                for u in udatasets:  # filter out collocated data files
                    if catalog_rms == u.split('/')[-1].split('_20')[0][15:]:
                        datasets.append(u)
                        deployments.append(int(u.split('/')[-1].split('_')[0][-4:]))
        deployments = np.unique(deployments).tolist()
        fdatasets = []
        if preferred_only == 'yes':
            # get the preferred stream information
            ps_df, n_streams = cf.get_preferred_stream_info(r)
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
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        for dep in deployments:
            if deployment_num is not None:
                if dep is not deployment_num:
                    print('\nskipping deployment {}'.format(dep))
                    continue
            rdatasets = [s for s in fdatasets_sel if 'deployment%04d' %dep in s]
            rdatasets.sort()
            if len(rdatasets) > 0:
                sci_vars_dict = {}
                # rdatasets = rdatasets[0:2]  #### for testing
                for i in range(len(rdatasets)):
                    ds = xr.open_dataset(rdatasets[i], mask_and_scale=False)
                    ds = ds.swap_dims({'obs': 'time'})
                    print('\nAppending data from {}: file {} of {}'.format('deployment%04d' % dep, i + 1, len(rdatasets)))

                    array = r[0:2]
                    subsite = r.split('-')[0]

                    if start_time is not None and end_time is not None:
                        ds = ds.sel(time=slice(start_time, end_time))
                        if len(ds['time'].values) == 0:
                            print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                            continue
                        stime = start_time.strftime('%Y-%m-%d')
                        etime = end_time.strftime('%Y-%m-%d')
                        ext = stime + 'to' + etime  # .join((ds0_method, ds1_method
                        save_dir_profile = os.path.join(sDir, array, subsite, r, 'profile_plots',
                                                        'deployment%04d' % dep, ext)
                        save_dir_xsection = os.path.join(sDir, array, subsite, r, 'xsection_plots',
                                                         'deployment%04d' % dep, ext)
                    else:
                        save_dir_profile = os.path.join(sDir, array, subsite, r, 'profile_plots', 'deployment%04d' % dep)
                        save_dir_xsection = os.path.join(sDir, array, subsite, r, 'xsection_plots', 'deployment%04d' % dep)

                    if len(sci_vars_dict) == 0:
                        fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(rdatasets[0])
                        sci_vars = cf.return_science_vars(stream)
                        if 'CTDPF' not in r:
                            sci_vars.append('int_ctd_pressure')
                        sci_vars.append('time')
                        sci_vars = list(np.unique(sci_vars))

                        # initialize the dictionary
                        for sci_var in sci_vars:
                            if sci_var == 'time':
                                sci_vars_dict.update(
                                    {sci_var: dict(values=np.array([], dtype=np.datetime64), units=[], fv=[])})
                            else:
                                sci_vars_dict.update({sci_var: dict(values=np.array([]), units=[], fv=[])})

                    # append data for the deployment into the dictionary
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
                                vv_data = vv.values
                                try:
                                    vv_data[vv_data == vv._FillValue] = np.nan  # turn fill values to nans
                                except ValueError:
                                    print('')
                        except AttributeError:
                            print('')

                        if len(vv.dims) > 1:
                            print('Skipping plot: variable has >1 dimension')
                        else:
                            sci_vars_dict[s_v]['values'] = np.append(sci_vars_dict[s_v]['values'], vv.values)

                # plot after appending all data into one file
                data_start = pd.to_datetime(min(sci_vars_dict['time']['values'])).strftime('%Y-%m-%dT%H:%M:%S')
                data_stop = pd.to_datetime(max(sci_vars_dict['time']['values'])).strftime('%Y-%m-%dT%H:%M:%S')
                time1 = sci_vars_dict['time']['values']
                ds_lat1 = np.empty(np.shape(time1))
                ds_lon1 = np.empty(np.shape(time1))

                # define pressure variable
                try:
                    pname = 'seawater_pressure'
                    press = sci_vars_dict[pname]
                except KeyError:
                    pname = 'int_ctd_pressure'
                    press = sci_vars_dict[pname]
                y1 = press['values']
                try:
                    y_units = press['units'][0]
                except IndexError:
                    y_units = ''

                for sv in sci_vars_dict.keys():
                    print('')
                    print(sv)
                    if sv not in ['seawater_pressure', 'int_ctd_pressure', 'time']:
                        z1 = sci_vars_dict[sv]['values']
                        fv = sci_vars_dict[sv]['fv'][0]
                        sv_units = sci_vars_dict[sv]['units'][0]

                        # Check if the array is all NaNs
                        if sum(np.isnan(z1)) == len(z1):
                            print('Array of all NaNs - skipping plot.')
                            continue

                        # Check if the array is all fill values
                        elif len(z1[z1 != fv]) == 0:
                            print('Array of all fill values - skipping plot.')
                            continue

                        else:
                            # remove unreasonable pressure data (e.g. for surface piercing profilers)
                            if zdbar:
                                po_ind = (0 < y1) & (y1 < zdbar)
                                tm = time1[po_ind]
                                y = y1[po_ind]
                                z = z1[po_ind]
                                ds_lat = ds_lat1[po_ind]
                                ds_lon = ds_lon1[po_ind]
                            else:
                                tm = time1
                                y = y1
                                z = z1
                                ds_lat = ds_lat1
                                ds_lon = ds_lon1

                            # reject erroneous data
                            dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max, lat, lon = \
                                cf.reject_erroneous_data(r, sv, tm, y, z, fv, ds_lat, ds_lon)

                            # get rid of 0.0 data
                            # if sv == 'salinity':
                            #     ind = ndata > 20
                            # elif sv == 'density':
                            #     ind = ndata > 1010
                            # elif sv == 'conductivity':
                            #     ind = ndata > 2
                            # else:
                            #     ind = ndata > 0
                            # if sv == 'sci_flbbcd_chlor_units':
                            #     ind = ndata < 7.5
                            # elif sv == 'sci_flbbcd_cdom_units':
                            #     ind = ndata < 25
                            # else:
                            #     ind = ndata > 0.0

                            if 'CTD' in r:
                                ind = zpressure > 0.0
                            else:
                                ind = ndata > 0.0

                            lenzero = np.sum(~ind)
                            dtime = dtime[ind]
                            zpressure = zpressure[ind]
                            ndata = ndata[ind]
                            if ds_lat is not None and ds_lon is not None:
                                lat = lat[ind]
                                lon = lon[ind]
                            else:
                                lat = None
                                lon = None

                            if len(dtime) > 0:
                                # reject time range from data portal file export
                                t_portal, z_portal, y_portal, lat_portal, lon_portal = \
                                    cf.reject_timestamps_dataportal(subsite, r, dtime, zpressure, ndata, lat, lon)

                                print('removed {} data points using visual inspection of data'.format(
                                    len(ndata) - len(z_portal)))

                                # create data groups
                                # if len(y_portal) > 0:
                                #     columns = ['tsec', 'dbar', str(sv)]
                                #     min_r = int(round(np.nanmin(y_portal) - zcell_size))
                                #     max_r = int(round(np.nanmax(y_portal) + zcell_size))
                                #     ranges = list(range(min_r, max_r, zcell_size))
                                #
                                #     groups, d_groups = gt.group_by_depth_range(t_portal, y_portal, z_portal, columns, ranges)
                                #
                                #     if 'scatter' in sv:
                                #         n_std = None  # to use percentile
                                #     else:
                                #         n_std = n_std
                                #
                                #     #  get percentile analysis for printing on the profile plot
                                #     y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex = cf.reject_timestamps_in_groups(
                                #         groups, d_groups, n_std, inpercentile)

                            """
                            Plot all data
                            """
                            if len(time1) > 0:
                                cf.create_dir(save_dir_profile)
                                cf.create_dir(save_dir_xsection)
                                sname = '-'.join((r, method, sv))
                                sfileall = '_'.join(('all_data', sname, pd.to_datetime(time1.min()).strftime('%Y%m%d')))
                                tm0 = pd.to_datetime(time1.min()).strftime('%Y-%m-%dT%H:%M:%S')
                                tm1 = pd.to_datetime(time1.max()).strftime('%Y-%m-%dT%H:%M:%S')
                                title = ' '.join((deployment, refdes, method)) + '\n' + tm0 + ' to ' + tm1
                                if 'SPKIR' in r:
                                    title = title + '\nWavelength = 510 nm'

                                '''
                                profile plot
                                '''
                                xlabel = sv + " (" + sv_units + ")"
                                ylabel = pname + " (" + y_units + ")"
                                clabel = 'Time'

                                fig, ax = pf.plot_profiles(z1, y1, time1, ylabel, xlabel, clabel, stdev=None)

                                ax.set_title(title, fontsize=9)
                                fig.tight_layout()
                                pf.save_fig(save_dir_profile, sfileall)

                                '''
                                xsection plot
                                '''
                                clabel = sv + " (" + sv_units + ")"
                                ylabel = pname + " (" + y_units + ")"

                                fig, ax, bar = pf.plot_xsection(subsite, time1, y1, z1, clabel, ylabel, t_eng=None,
                                                                m_water_depth=None, inpercentile=None, stdev=None)

                                if fig:
                                    ax.set_title(title, fontsize=9)
                                    fig.tight_layout()
                                    pf.save_fig(save_dir_xsection, sfileall)

                            """
                            Plot cleaned-up data
                            """
                            # if len(dtime) > 0:
                            #     if len(y_portal) > 0:
                            #         sfile = '_'.join(('rm_erroneous_data', sname, pd.to_datetime(t_portal.min()).strftime('%Y%m%d')))
                            #         t0 = pd.to_datetime(t_portal.min()).strftime('%Y-%m-%dT%H:%M:%S')
                            #         t1 = pd.to_datetime(t_portal.max()).strftime('%Y-%m-%dT%H:%M:%S')
                            #         title = ' '.join((deployment, refdes, method)) + '\n' + t0 + ' to ' + t1
                            #         if 'SPKIR' in r:
                            #             title = title + '\nWavelength = 510 nm'
                            #
                            #         '''
                            #         profile plot
                            #         '''
                            #         xlabel = sv + " (" + sv_units + ")"
                            #         ylabel = pname + " (" + y_units + ")"
                            #         clabel = 'Time'
                            #
                            #         fig, ax = pf.plot_profiles(z_portal, y_portal, t_portal, ylabel, xlabel, clabel, stdev=None)
                            #
                            #         ax.set_title(title, fontsize=9)
                            #         ax.plot(n_avg, y_avg, '-k')
                            #         ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                            #         if inpercentile:
                            #             leg_text = (
                            #                 'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                            #                 '{} unreasonable values'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                            #                 '\nexcluded {} suspect data points when inspected visually'.format(
                            #                     len(ndata) - len(z_portal)) +
                            #                 '\n(black) data average in {} dbar segments'.format(zcell_size) +
                            #                 '\n(magenta) {} percentile envelope in {} dbar segments'.format(
                            #                     int(100 - inpercentile * 2), zcell_size),)
                            #         elif n_std:
                            #             leg_text = (
                            #                 'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                            #                 '{} unreasonable values'.format(lenfv, lennan, lenev, lengr, global_min, global_max,
                            #                                   lenzero) +
                            #                 '\nexcluded {} suspect data points when inspected visually'.format(
                            #                     len(ndata) - len(z_portal)) +
                            #                 '\n(black) data average in {} dbar segments'.format(zcell_size) +
                            #                 '\n(magenta) +/- {} SD envelope in {} dbar segments'.format(
                            #                     int(n_std), zcell_size),)
                            #         ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            #         fig.tight_layout()
                            #         pf.save_fig(save_dir_profile, sfile)
                            #
                            #         '''
                            #         xsection plot
                            #         '''
                            #         clabel = sv + " (" + sv_units + ")"
                            #         ylabel = pname + " (" + y_units + ")"
                            #
                            #         # plot non-erroneous data
                            #         fig, ax, bar = pf.plot_xsection(subsite, t_portal, y_portal, z_portal, clabel, ylabel,
                            #                                         t_eng=None, m_water_depth=None, inpercentile=None,
                            #                                         stdev=None)
                            #
                            #         ax.set_title(title, fontsize=9)
                            #         leg_text = (
                            #             'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                            #             '{} unreasonable values'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                            #             '\nexcluded {} suspect data points when inspected visually'.format(
                            #                 len(ndata) - len(z_portal)),
                            #         )
                            #         ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            #         fig.tight_layout()
                            #         pf.save_fig(save_dir_xsection, sfile)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

    """
    define time range: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    """
    start_time = None  # dt.datetime(2014, 12, 1)
    end_time = None  # dt.datetime(2015, 5, 2)

    '''
    define filters standard deviation, percentile
    '''
    n_std = None
    inpercentile = 5

    '''
    define the depth cell_size for data grouping 
    '''
    zcell_size = 5

    zdbar = None  # remove data below this depth for analysis and plotting

    ''''
    define deployment number and indicate if only the preferred data should be plotted
    '''
    deployment_num = None
    preferred_only = 'yes'  # options: 'yes', 'no'

    '''
    output directory, and data files URL location
    '''
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20191003T163109205Z-CE04OSPS-SF01B-2A-CTDPFA107-streamed-ctdpf_sbe43_sample/catalog.html']

    '''
    call in main function with the above attributes
    '''
    main(url_list, sDir, deployment_num, start_time, end_time, preferred_only, n_std, inpercentile, zcell_size, zdbar)
