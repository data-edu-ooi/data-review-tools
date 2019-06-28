#!/usr/bin/env python
"""
Created on 11/12/2018
@author Lori Garzio
@brief Calculate final statistics for science variables in a dataset. Long Names and units for science variables must
be the same among methods.
sDir: location to save summary output
url: list of THREDDs urls containing .nc files to analyze.
"""

import os
import itertools
import pandas as pd
import numpy as np
import datetime as dt
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd


def index_dataset(refdes, var_name, var_data, fv):
    n_nan = np.sum(np.isnan(var_data))
    n_fv = np.sum(var_data == fv)
    [g_min, g_max] = cf.get_global_ranges(refdes, var_name)
    if g_min is not None and g_max is not None:
        dataind = (~np.isnan(var_data)) & (var_data != fv) & (var_data >= g_min) & (var_data <= g_max)
        n_grange = np.sum((var_data < g_min) | (var_data > g_max))
    else:
        dataind = (~np.isnan(var_data)) & (var_data != fv)
        n_grange = 'no global ranges'

    return [dataind, g_min, g_max, n_nan, n_fv, n_grange]


def index_dataset_2d(refdes, var_name, var_data, fv):
    [g_min, g_max] = cf.get_global_ranges(refdes, var_name)
    fdata = dict()
    n_nan = []
    n_fv = []
    n_grange = []
    for i in range(len(var_data)):
        vd = var_data[i]
        n_nani = np.sum(np.isnan(vd))

        # convert fill values to nans
        vd[vd == fv] = np.nan
        n_fvi = np.sum(np.isnan(vd)) - n_nani

        if g_min is not None and g_max is not None:
            vd[vd < g_min] = np.nan
            vd[vd > g_max] = np.nan
            n_grangei = np.sum(np.isnan(vd) - n_fvi - n_nani)
        else:
            n_grangei = 'no global ranges'

        fdata.update({i: vd})
        n_nan.append(int(n_nani))
        n_fv.append(int(n_fvi))
        try:
            n_grange.append(int(n_grangei))
        except ValueError:
            n_grange.append(n_grangei)

    return [fdata, g_min, g_max, n_nan, n_fv, n_grange]


def main(sDir, plotting_sDir, url_list, sd_calc):
    dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
    drn = dr.loc[dr.type == 'exclusion']
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        fdatasets = []
        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)
        pms = []
        for index, row in ps_df.iterrows():
            for ii in range(n_streams):
                try:
                    rms = '-'.join((r, row[ii]))
                    pms.append(row[ii])
                except TypeError:
                    continue
                for dd in datasets:
                    spl = dd.split('/')[-2].split('-')
                    catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                    fdeploy = dd.split('/')[-1].split('_')[0]
                    if rms == catalog_rms and fdeploy == row['deployment']:
                        fdatasets.append(dd)

        main_sensor = r.split('-')[-1]
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        # find time ranges to exclude from analysis for data review database
        subsite = r.split('-')[0]
        subsite_node = '-'.join((subsite, r.split('-')[1]))

        drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]
        et = []
        for i, row in drne.iterrows():
            sdate = cf.format_dates(row.start_date)
            edate = cf.format_dates(row.end_date)
            et.append([sdate, edate])

        # get science variable long names from the Data Review Database
        stream_sci_vars = cd.sci_var_long_names(r)

        # check if the science variable long names are the same for each stream
        sci_vars_dict = cd.sci_var_long_names_check(stream_sci_vars)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        sci_vars_dict, pressure_unit, pressure_name = cd.append_science_data(ps_df, n_streams, r, fdatasets_sel, sci_vars_dict, et)

        # analyze combined dataset
        print('\nAnalyzing combined dataset and writing summary file')

        array = subsite[0:2]
        save_dir = os.path.join(sDir, array, subsite)
        cf.create_dir(save_dir)

        rows = []
        if ('FLM' in r) and ('CTDMO' in r):  # calculate Flanking Mooring CTDMO stats based on pressure
            headers = ['common_stream_name', 'preferred_methods_streams', 'deployments', 'long_name', 'units', 't0',
                       't1', 'fill_value', 'global_ranges', 'n_all', 'press_min_max', 'n_excluded_forpress',
                       'n_nans', 'n_fillvalues', 'n_grange', 'define_stdev', 'n_outliers', 'n_stats', 'mean', 'min',
                       'max', 'stdev', 'note']
        else:
            headers = ['common_stream_name', 'preferred_methods_streams', 'deployments', 'long_name', 'units', 't0',
                       't1', 'fill_value', 'global_ranges', 'n_all', 'n_nans', 'n_fillvalues', 'n_grange',
                       'define_stdev', 'n_outliers', 'n_stats', 'mean', 'min', 'max', 'stdev']

        for m, n in sci_vars_dict.items():
            print('\nSTREAM: ', m)
            if m == 'common_stream_placeholder':
                m = 'science_data_stream'
            if m == 'metbk_hourly':  # don't calculate ranges for metbk_hourly
                continue

            if ('FLM' in r) and ('CTDMO' in r):  # calculate Flanking Mooring CTDMO stats based on pressure
                # index the pressure variable to filter and calculate stats on the rest of the variables
                sv_press = 'Seawater Pressure'
                vinfo_press = n['vars'][sv_press]

                # first, index where data are nans, fill values, and outside of global ranges
                fv_press = list(np.unique(vinfo_press['fv']))[0]
                pdata = vinfo_press['values']

                [pind, __, __, __, __, __] = index_dataset(r, vinfo_press['var_name'], pdata, fv_press)

                pdata_filtered = pdata[pind]
                [__, pmean, __, __, psd, __] = cf.variable_statistics(pdata_filtered, None)

                # index of pressure = average of all 'valid' pressure data +/- 1 SD
                ipress_min = pmean - psd
                ipress_max = pmean + psd
                ind_press = (pdata >= ipress_min) & (pdata <= ipress_max)

                # calculate stats for all variables
                print('\nPARAMETERS:')
                for sv, vinfo in n['vars'].items():
                    print(sv)
                    fv_lst = np.unique(vinfo['fv']).tolist()
                    if len(fv_lst) == 1:
                        fill_value = fv_lst[0]
                    else:
                        print('No unique fill value for {}'.format(sv))

                    lunits = np.unique(vinfo['units']).tolist()
                    n_all = len(vinfo['t'])

                    # filter data based on pressure index
                    t_filtered = vinfo['t'][ind_press]
                    data_filtered = vinfo['values'][ind_press]
                    deploy_filtered = vinfo['deployments'][ind_press]

                    n_excluded = n_all - len(t_filtered)

                    [dataind, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset(r, vinfo['var_name'], data_filtered, fill_value)

                    t_final = t_filtered[dataind]
                    data_final = data_filtered[dataind]
                    deploy_final = deploy_filtered[dataind]

                    t0 = pd.to_datetime(min(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(max(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                    deploy = list(np.unique(deploy_final))
                    deployments = [int(dd) for dd in deploy]

                    if len(data_final) > 1:
                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_final, sd_calc)
                    else:
                        mean = None
                        vmin = None
                        vmax = None
                        sd = None
                        n_stats = None

                    note = 'restricted stats calculation to data points where pressure is within defined ranges' \
                           ' (average of all pressure data +/- 1 SD)'
                    rows.append([m, list(np.unique(pms)), deployments, sv, lunits, t0, t1, fv_lst, [g_min, g_max],
                                 n_all, [round(ipress_min, 2), round(ipress_max, 2)], n_excluded, n_nan, n_fv, n_grange,
                                 sd_calc, num_outliers, n_stats, mean, vmin, vmax, sd, note])

                    # plot CTDMO data used for stats
                    psave_dir = os.path.join(plotting_sDir, array, subsite, r, 'timeseries_plots_stats')
                    cf.create_dir(psave_dir)

                    dr_data = cf.refdes_datareview_json(r)
                    deployments = []
                    end_times = []
                    for index, row in ps_df.iterrows():
                        deploy = row['deployment']
                        deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
                        deployments.append(int(deploy[-4:]))
                        end_times.append(pd.to_datetime(deploy_info['stop_date']))

                    sname = '-'.join((r, sv))
                    fig, ax = pf.plot_timeseries_all(t_final, data_final, sv, lunits[0], stdev=None)
                    ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                 fontsize=8)
                    for etimes in end_times:
                        ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                    pf.save_fig(psave_dir, sname)

                    if sd_calc:
                        sname = '-'.join((r, sv, 'rmoutliers'))
                        fig, ax = pf.plot_timeseries_all(t_final, data_final, sv, lunits[0], stdev=sd_calc)
                        ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                     fontsize=8)
                        for etimes in end_times:
                            ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                        pf.save_fig(psave_dir, sname)

            else:
                if not sd_calc:
                    sdcalc = None

                print('\nPARAMETERS: ')
                for sv, vinfo in n['vars'].items():
                    print(sv)

                    fv_lst = np.unique(vinfo['fv']).tolist()
                    if len(fv_lst) == 1:
                        fill_value = fv_lst[0]
                    else:
                        print(fv_lst)
                        print('No unique fill value for {}'.format(sv))

                    lunits = np.unique(vinfo['units']).tolist()

                    t = vinfo['t']
                    [g_min, g_max] = cf.get_global_ranges(r, vinfo['var_name'])
                    if len(t) > 1:
                        data = vinfo['values']
                        n_all = len(t)

                        if 'SPKIR' in r or 'presf_abc_wave_burst' in m:
                            if 'SPKIR' in r:
                                [dd_data, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset_2d(r, 'spkir_abj_cspp_downwelling_vector',
                                                                                                  data, fill_value)
                            else:
                                [dd_data, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset_2d(r,'presf_wave_burst_pressure',
                                                                                                  data, fill_value)
                            t_final = t
                            t0 = pd.to_datetime(min(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                            t1 = pd.to_datetime(max(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                            deploy_final = vinfo['deployments']
                            deploy = list(np.unique(deploy_final))
                            deployments = [int(dd) for dd in deploy]

                            num_outliers = []
                            mean = []
                            vmin = []
                            vmax = []
                            sd = []
                            n_stats = []
                            for i in range(len(dd_data)):
                                dd = data[i]
                                # drop nans before calculating stats
                                dd = dd[~np.isnan(dd)]
                                [num_outliersi, meani, vmini, vmaxi, sdi, n_statsi] = cf.variable_statistics(dd, sd_calc)
                                num_outliers.append(num_outliersi)
                                mean.append(meani)
                                vmin.append(vmini)
                                vmax.append(vmaxi)
                                sd.append(sdi)
                                n_stats.append(n_statsi)

                        else:
                            if type(vinfo['values']) == dict:  # if the variable is a 2D array
                                [dd_data, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset_2d(r, vinfo['var_name'],
                                                                                                  data, fill_value)
                                t_final = t
                                t0 = pd.to_datetime(min(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                                t1 = pd.to_datetime(max(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                                deploy_final = vinfo['deployments']
                                deploy = list(np.unique(deploy_final))
                                deployments = [int(dd) for dd in deploy]

                                num_outliers = []
                                mean = []
                                vmin = []
                                vmax = []
                                sd = []
                                n_stats = []
                                for i in range(len(dd_data)):
                                    dd = data[i]
                                    # drop nans before calculating stats
                                    dd = dd[~np.isnan(dd)]
                                    [num_outliersi, meani, vmini, vmaxi, sdi, n_statsi] = cf.variable_statistics(dd,
                                                                                                                 sd_calc)
                                    num_outliers.append(num_outliersi)
                                    mean.append(meani)
                                    vmin.append(vmini)
                                    vmax.append(vmaxi)
                                    sd.append(sdi)
                                    n_stats.append(n_statsi)
                            else:
                                [dataind, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset(r, vinfo['var_name'],
                                                                                               data, fill_value)
                                t_final = t[dataind]
                                if len(t_final) > 0:
                                    t0 = pd.to_datetime(min(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                                    t1 = pd.to_datetime(max(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                                    data_final = data[dataind]
                                    # if sv == 'Dissolved Oxygen Concentration':
                                    #     xx = (data_final > 0) & (data_final < 400)
                                    #     data_final = data_final[xx]
                                    #     t_final = t_final[xx]
                                    # if sv == 'Seawater Conductivity':
                                    #     xx = (data_final > 1) & (data_final < 400)
                                    #     data_final = data_final[xx]
                                    #     t_final = t_final[xx]
                                    deploy_final = vinfo['deployments'][dataind]
                                    deploy = list(np.unique(deploy_final))
                                    deployments = [int(dd) for dd in deploy]

                                    if len(data_final) > 1:
                                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_final, sd_calc)
                                    else:
                                        sdcalc = None
                                        num_outliers = None
                                        mean = None
                                        vmin = None
                                        vmax = None
                                        sd = None
                                        n_stats = None
                                else:
                                    sdcalc = None
                                    num_outliers = None
                                    mean = None
                                    vmin = None
                                    vmax = None
                                    sd = None
                                    n_stats = None
                                    deployments = None
                                    t0 = None
                                    t1 = None
                    else:
                        sdcalc = None
                        num_outliers = None
                        mean = None
                        vmin = None
                        vmax = None
                        sd = None
                        n_stats = None
                        deployments = None
                        t0 = None
                        t1 = None
                        t_final = []
                        n_all = None
                        n_nan = None
                        n_fv = None
                        n_grange = None

                    if sd_calc:
                        print_sd = sd_calc
                    else:
                        print_sd = sdcalc

                    rows.append([m, list(np.unique(pms)), deployments, sv, lunits, t0, t1, fv_lst, [g_min, g_max],
                                 n_all, n_nan, n_fv, n_grange, print_sd, num_outliers, n_stats, mean, vmin, vmax, sd])

                    if len(t_final) > 0:
                        # plot data used for stats
                        psave_dir = os.path.join(plotting_sDir, array, subsite, r, 'timeseries_reviewed_datarange')
                        cf.create_dir(psave_dir)

                        dr_data = cf.refdes_datareview_json(r)
                        deployments = []
                        end_times = []
                        for index, row in ps_df.iterrows():
                            deploy = row['deployment']
                            deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
                            deployments.append(int(deploy[-4:]))
                            end_times.append(pd.to_datetime(deploy_info['stop_date']))

                        sname = '-'.join((r, sv))

                        # plot hourly averages for streaming data
                        if 'streamed' in sci_vars_dict[list(sci_vars_dict.keys())[0]]['ms'][0]:
                            sname = '-'.join((sname, 'hourlyavg'))
                            df = pd.DataFrame({'dfx': t_final, 'dfy': data_final})
                            dfr = df.resample('H', on='dfx').mean()

                            # Plot all data
                            fig, ax = pf.plot_timeseries_all(dfr.index, dfr['dfy'], sv, lunits[0], stdev=None)
                            ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                         fontsize=8)
                            for etimes in end_times:
                                ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                            pf.save_fig(psave_dir, sname)
                            
                            if sd_calc:
                                sname = '-'.join((sname, 'hourlyavg_rmoutliers'))
                                fig, ax = pf.plot_timeseries_all(dfr.index, dfr['dfy'], sv, lunits[0], stdev=sd_calc)
                                ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                             fontsize=8)
                                for etimes in end_times:
                                    ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                                pf.save_fig(psave_dir, sname)
                                
                        elif 'SPKIR' in r:
                            fig, ax = pf.plot_spkir(t_final, dd_data, sv, lunits[0])
                            ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                         fontsize=8)
                            for etimes in end_times:
                                ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                            pf.save_fig(psave_dir, sname)

                            # plot each wavelength
                            wavelengths = ['412nm', '443nm', '490nm', '510nm', '555nm', '620nm', '683nm']
                            for wvi in range(len(dd_data)):
                                fig, ax = pf.plot_spkir_wv(t_final, dd_data[wvi], sv, lunits[0], wvi)
                                ax.set_title(
                                    (r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                    fontsize=8)
                                for etimes in end_times:
                                    ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                                snamewvi = '-'.join((sname, wavelengths[wvi]))
                                pf.save_fig(psave_dir, snamewvi)
                        elif 'presf_abc_wave_burst' in m:
                            fig, ax = pf.plot_presf_2d(t_final, dd_data, sv, lunits[0])
                            ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                         fontsize=8)
                            for etimes in end_times:
                                ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                            snamewave = '-'.join((sname, m))
                            pf.save_fig(psave_dir, snamewave)

                        else:  # plot all data if not streamed or 2D
                            if type(vinfo['values']) != dict:  # if the variable is not a 2D array
                                fig, ax = pf.plot_timeseries_all(t_final, data_final, sv, lunits[0], stdev=None)
                                ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                             fontsize=8)
                                for etimes in end_times:
                                    ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                                pf.save_fig(psave_dir, sname)

                                if sd_calc:
                                    sname = '-'.join((r, sv, 'rmoutliers'))
                                    fig, ax = pf.plot_timeseries_all(t_final, data_final, sv, lunits[0], stdev=sd_calc)
                                    ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                                 fontsize=8)
                                    for etimes in end_times:
                                        ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                                    pf.save_fig(psave_dir, sname)

        fsum = pd.DataFrame(rows, columns=headers)
        fsum.to_csv('{}/{}_data_ranges.csv'.format(save_dir, r), index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/github/data-review-tools/data_review/data_ranges'
    plotting_sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    sd_calc = 1  # number of standard deviations for outlier calculation. options: int or None

    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190514T200839-CE04OSBP-LJ01C-10-PHSEND107-streamed-phsen_data_record/catalog.html']

    main(sDir, plotting_sDir, url_list, sd_calc)
