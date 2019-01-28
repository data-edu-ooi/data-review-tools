#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for all deployments of a reference
designator using the preferred stream for each deployment (plots may contain data from different delivery methods if
data from the 'preferred' delivery method isn't available for a deployment: 1) plot all data, 2) plot data, omitting
outliers beyond 5 standard deviations.
"""

import os
import pandas as pd
import itertools
import numpy as np
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def var_long_names(refdes):
    # get science variable long names from the Data Review Database
    stream_vars_dict = dict()
    dr = cf.refdes_datareview_json(refdes)
    for x in dr['instrument']['data_streams']:
        dr_ms = '-'.join((x['method'], x['stream_name']))
        sci_vars = dict()
        for y in x['stream']['parameters']:
            if (y['data_product_type'] == 'Science Data') or (y['data_product_type'] == 'Unprocessed Data'):
                sci_vars.update({y['display_name']: dict(db_units=y['unit'], var_name=y['name'])})
        if len(sci_vars) > 0:
            stream_vars_dict.update({dr_ms: sci_vars})
    return stream_vars_dict


def main(sDir, url_list, start_time, end_time):
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
        for index, row in ps_df.iterrows():
            for ii in range(n_streams):
                rms = '-'.join((r, row[ii]))
                for dd in datasets:
                    spl = dd.split('/')[-2].split('-')
                    catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                    fdeploy = dd.split('/')[-1].split('_')[0]
                    if rms == catalog_rms and fdeploy == row['deployment']:
                        fdatasets.append(dd)

        main_sensor = r.split('-')[-1]
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        # get science variable long names from the Data Review Database
        #stream_sci_vars = cd.sci_var_long_names(r)
        stream_vars = var_long_names(r)

        # check if the science variable long names are the same for each stream and initialize empty arrays
        sci_vars_dict = cd.sci_var_long_names_check(stream_vars)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        et = []
        sci_vars_dict = cd.append_science_data(ps_df, n_streams, r, fdatasets_sel, sci_vars_dict, et, start_time, end_time)

        # get end times of deployments
        dr_data = cf.refdes_datareview_json(r)
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        subsite = r.split('-')[0]
        array = subsite[0:2]
        save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_plots_preferred_all')
        cf.create_dir(save_dir)

        print('\nPlotting data')
        for m, n in sci_vars_dict.items():
            for sv, vinfo in n['vars'].items():
                print(sv)
                if len(vinfo['t']) < 1:
                    print('no variable data to plot')
                else:
                    sv_units = vinfo['units'][0]
                    sv_name = vinfo['var_name']
                    t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                    x = vinfo['t']
                    y = vinfo['values']

                    # reject NaNs and values of 0.0
                    nan_ind = (~np.isnan(y)) & (y != 0.0)
                    x_nonan = x[nan_ind]
                    y_nonan = y[nan_ind]

                    # reject fill values
                    fv_ind = y_nonan != vinfo['fv'][0]
                    x_nonan_nofv = x_nonan[fv_ind]
                    y_nonan_nofv = y_nonan[fv_ind]

                    # reject extreme values
                    Ev_ind = cf.reject_extreme_values(y_nonan_nofv)
                    y_nonan_nofv_nE = y_nonan_nofv[Ev_ind]
                    x_nonan_nofv_nE = x_nonan_nofv[Ev_ind]

                    # reject values outside global ranges:
                    global_min, global_max = cf.get_global_ranges(r, sv_name)
                    if any(e is None for e in [global_min, global_max]):
                        y_nonan_nofv_nE_nogr = y_nonan_nofv_nE
                        x_nonan_nofv_nE_nogr = x_nonan_nofv_nE
                    else:
                        gr_ind = cf.reject_global_ranges(y_nonan_nofv_nE, global_min, global_max)
                        y_nonan_nofv_nE_nogr = y_nonan_nofv_nE[gr_ind]
                        x_nonan_nofv_nE_nogr = x_nonan_nofv_nE[gr_ind]

                    if len(y_nonan_nofv) > 0:
                        if m == 'common_stream_placeholder':
                            sname = '-'.join((r, sv))
                        else:
                            sname = '-'.join((r, m, sv))

                        # Plot all data
                        fig, ax = pf.plot_timeseries_all(x_nonan_nofv, y_nonan_nofv, sv, sv_units, stdev=None)
                        ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                     fontsize=8)
                        for etimes in end_times:
                            ax.axvline(x=etimes,  color='b', linestyle='--', linewidth=.6)
                        # if not any(e is None for e in [global_min, global_max]):
                        #     ax.axhline(y=global_min, color='r', linestyle='--', linewidth=.6)
                        #     ax.axhline(y=global_max, color='r', linestyle='--', linewidth=.6)
                        # else:
                        #     maxpoint = x[np.argmax(y_nonan_nofv)], max(y_nonan_nofv)
                        #     ax.annotate('No Global Ranges', size=8,
                        #                 xy=maxpoint, xytext=(5, 5), textcoords='offset points')
                        pf.save_fig(save_dir, sname)

                        # Plot data with outliers removed
                        fig, ax = pf.plot_timeseries_all(x_nonan_nofv_nE_nogr, y_nonan_nofv_nE_nogr, sv, sv_units,
                                                         stdev=5)
                        ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                     fontsize=8)
                        for etimes in end_times:
                            ax.axvline(x=etimes,  color='b', linestyle='--', linewidth=.6)
                        # if not any(e is None for e in [global_min, global_max]):
                        #     ax.axhline(y=global_min, color='r', linestyle='--', linewidth=.6)
                        #     ax.axhline(y=global_max, color='r', linestyle='--', linewidth=.6)
                        # else:
                        #     maxpoint = x[np.argmax(y_nonan_nofv_nE_nogr)], max(y_nonan_nofv_nE_nogr)
                        #     ax.annotate('No Global Ranges', size=8,
                        #                 xy=maxpoint, xytext=(5, 5), textcoords='offset points')

                        sfile = '_'.join((sname, 'rmoutliers'))
                        pf.save_fig(save_dir, sfile)



if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  #for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = [
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163751-CE09OSPM-WFP01-02-DOFSTK000-recovered_wfp-dofst_k_wfp_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163824-CE09OSPM-WFP01-02-DOFSTK000-telemetered-dofst_k_wfp_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163408-CE06ISSM-RID16-03-DOSTAD000-recovered_host-dosta_abcdjm_ctdbp_dcl_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163419-CE06ISSM-RID16-03-DOSTAD000-recovered_inst-dosta_abcdjm_ctdbp_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163558-CE06ISSM-RID16-03-DOSTAD000-telemetered-dosta_abcdjm_ctdbp_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163845-CE09OSSM-RID27-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163907-CE09OSSM-RID27-04-DOSTAD000-telemetered-dosta_abcdjm_dcl_instrument/catalog.html']

    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    main(sDir, url_list)
