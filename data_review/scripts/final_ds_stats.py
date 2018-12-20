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


def format_dates(dd):
    fd = dt.datetime.strptime(dd.replace(',', ''), '%m/%d/%y %I:%M %p')
    fd2 = dt.datetime.strftime(fd, '%Y-%m-%dT%H:%M:%S')
    return fd2


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def index_dataset(refdes, var_name, var_data, fv):
    n_nan = np.sum(np.isnan(var_data))
    n_fv = np.sum(var_data == fv)

    [g_min, g_max] = cf.get_global_ranges(refdes, var_name)
    if g_min is not None and g_max is not None:
        dataind = (~np.isnan(var_data)) & (var_data != fv) & (var_data >= g_min) & (var_data <= g_max)
        n_grange = np.sum((var_data < g_min) & (var_data > g_max))
    else:
        dataind = (~np.isnan(var_data)) & (var_data == fv)
        n_grange = 'no global ranges'

    return [dataind, g_min, g_max, n_nan, n_fv, n_grange]


def main(sDir, plotting_sDir, url_list):
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
                rms = '-'.join((r, row[ii]))
                pms.append(row[ii])
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
            sdate = format_dates(row.start_date)
            edate = format_dates(row.end_date)
            et.append([sdate, edate])

        # get science variable long names from the Data Review Database
        stream_sci_vars = cd.sci_var_long_names(r)

        # check if the science variable long names are the same for each stream
        sci_vars_dict = cd.sci_var_long_names_check(stream_sci_vars)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        sci_vars_dict = cd.append_science_data(ps_df, n_streams, r, fdatasets_sel, sci_vars_dict, et)

        # analyze combined dataset
        print('\nAnalyzing combined dataset and writing summary file')

        array = subsite[0:2]
        save_dir = os.path.join(sDir, array, subsite)
        cf.create_dir(save_dir)

        for m, n in sci_vars_dict.items():
            print(m)
            if m == 'common_stream_placeholder':
                m = 'science_data_stream'

            if 'CTDMO' in r:
                headers = ['common_stream_name', 'preferred_methods_streams', 'deployments', 'long_name', 'units', 't0',
                           't1', 'fill_value', 'global_ranges', 'n_all', 'press_min_max', 'n_excluded_forpress',
                           'n_nans', 'n_fillvalues', 'n_grange', 'n_stats', 'mean', 'min', 'max', 'stdev', 'note']
                rows = []

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
                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_final, None)
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
                                 n_stats, mean, vmin, vmax, sd, note])

                    # plot CTDMO data used for stats
                    psave_dir = os.path.join(plotting_sDir, array, subsite, r, 'timeseries_plots_stats')
                    cf.create_dir(psave_dir)

                    dr_data = cf.refdes_datareview_json(r)
                    deployments = []
                    end_times = []
                    for index, row in ps_df.iterrows():
                        deploy = row['deployment']
                        deploy_info = get_deployment_information(dr_data, int(deploy[-4:]))
                        deployments.append(int(deploy[-4:]))
                        end_times.append(pd.to_datetime(deploy_info['stop_date']))

                    sname = '-'.join((r, sv))
                    fig, ax = pf.plot_timeseries_all(t_final, data_final, sv, lunits[0], stdev=None)
                    ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                 fontsize=8)
                    for etimes in end_times:
                        ax.axvline(x=etimes, color='k', linestyle='--', linewidth=.6)
                    pf.save_fig(psave_dir, sname)

            else:
                headers = ['common_stream_name', 'preferred_methods_streams', 'deployments', 'long_name', 'units', 't0',
                           't1', 'fill_value', 'global_ranges', 'n_all', 'n_nans', 'n_fillvalues', 'n_grange',
                           'define_stdev', 'n_outliers', 'n_stats', 'mean', 'min', 'max', 'stdev']
                rows = []
                for sv, vinfo in n['vars'].items():
                    print(sv)

                    fv_lst = np.unique(vinfo['fv']).tolist()
                    if len(fv_lst) == 1:
                        fill_value = fv_lst[0]
                    else:
                        print('No unique fill value for {}'.format(sv))

                    lunits = np.unique(vinfo['units']).tolist()

                    t = vinfo['t']
                    data = vinfo['values']
                    n_all = len(t)

                    [dataind, g_min, g_max, n_nan, n_fv, n_grange] = index_dataset(r, vinfo['var_name'], data, fill_value)

                    t_final = t[dataind]
                    t0 = pd.to_datetime(min(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(max(t_final)).strftime('%Y-%m-%dT%H:%M:%S')
                    data_final = data[dataind]
                    deploy_final = vinfo['deployments'][dataind]
                    deploy = list(np.unique(deploy_final))
                    deployments = [int(dd) for dd in deploy]

                    if len(data_final) > 1:
                        sd_calc = None  # number of standard deviations for outlier calculation. options: int or None
                        [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_final, sd_calc)
                    else:
                        sd_calc = None
                        num_outliers = None
                        mean = None
                        vmin = None
                        vmax = None
                        sd = None
                        n_stats = None

                    rows.append([m, list(np.unique(pms)), deployments, sv, lunits, t0, t1, fv_lst, [g_min, g_max],
                                 n_all, n_nan, n_fv, n_grange, sd_calc, num_outliers, n_stats, mean, vmin, vmax, sd])

        fsum = pd.DataFrame(rows, columns=headers)
        fsum.to_csv('{}/{}_final_stats.csv'.format(save_dir, r), index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/final_stats'
    plotting_sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022407-GI03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022421-GI03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022434-GI03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']

    main(sDir, plotting_sDir, url_list)
