#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for all deployments of a reference
designator using the preferred stream for each deployment (plots may contain data from different delivery methods if
data from the 'preferred' delivery method isn't available for a deployment: 1) plot all data, 2) plot data, omitting
outliers beyond 3 standard deviations.
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


def main(sDir, url_list):
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
        stream_sci_vars = cd.sci_var_long_names(r)

        # check if the science variable long names are the same for each stream and initialize empty arrays
        sci_vars_dict = cd.sci_var_long_names_check(stream_sci_vars)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        sci_vars_dict = cd.append_science_data(ps_df, n_streams, r, fdatasets_sel, sci_vars_dict)

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
        save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_plots_all')
        cf.create_dir(save_dir)

        print('\nPlotting data')
        for m, n in sci_vars_dict.items():
            for sv, vinfo in n['vars'].items():
                print(sv)
                sv_units = vinfo['units'][0]
                t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                x = vinfo['t']
                y = vinfo['values']

                # reject NaNs
                nan_ind = ~np.isnan(y)
                x_nonan = x[nan_ind]
                y_nonan = y[nan_ind]

                # reject fill values
                fv_ind = y_nonan != vinfo['fv'][0]
                x_nonan_nofv = x_nonan[fv_ind]
                y_nonan_nofv = y_nonan[fv_ind]

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
                        ax.axvline(x=etimes,  color='k', linestyle='--', linewidth=.6)
                    pf.save_fig(save_dir, sname)

                    # Plot data with outliers removed
                    fig, ax = pf.plot_timeseries_all(x_nonan_nofv, y_nonan_nofv, sv, sv_units, stdev=3)
                    ax.set_title((r + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                 fontsize=8)
                    for etimes in end_times:
                        ax.axvline(x=etimes,  color='k', linestyle='--', linewidth=.6)
                    sfile = '_'.join((sname, 'rmoutliers'))
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172034-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172050-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172104-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']
    main(sDir, url_list)
