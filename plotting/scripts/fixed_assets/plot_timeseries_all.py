#!/usr/bin/env python
"""
Created on Dec 14 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for all deployments of a reference
designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
"""

import os
import pandas as pd
import itertools
import numpy as np
import xarray as xr
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
        subsite = r.split('-')[0]
        array = subsite[0:2]

        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # get end times of deployments
        dr_data = cf.refdes_datareview_json(r)
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        # filter datasets
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        main_sensor = r.split('-')[-1]
        fdatasets = cf.filter_collocated_instruments(main_sensor, datasets)
        methodstream = []
        for f in fdatasets:
            methodstream.append('-'.join((f.split('/')[-2].split('-')[-2], f.split('/')[-2].split('-')[-1])))

        for ms in np.unique(methodstream):
            fdatasets_sel = [x for x in fdatasets if ms in x]
            save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_plots_all', ms.split('-')[0])
            cf.create_dir(save_dir)

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

            sci_vars_dict = cd.initialize_empty_arrays(stream_sci_vars_dict, ms)
            print('\nAppending data from files: {}'.format(ms))
            for fd in fdatasets_sel:
                ds = xr.open_dataset(fd, mask_and_scale=False)
                for var in list(sci_vars_dict[ms]['vars'].keys()):
                    sh = sci_vars_dict[ms]['vars'][var]
                    if ds[var].units == sh['db_units']:
                        if ds[var]._FillValue not in sh['fv']:
                            sh['fv'].append(ds[var]._FillValue)
                        if ds[var].units not in sh['units']:
                            sh['units'].append(ds[var].units)
                        tD = ds['time'].values
                        varD = ds[var].values
                        sh['t'] = np.append(sh['t'], tD)
                        sh['values'] = np.append(sh['values'], varD)

            print('\nPlotting data')
            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print(sv)
                    if len(vinfo['t']) < 1:
                        print('no variable data to plot')
                    else:
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

                        # reject extreme values
                        Ev_ind = cf.reject_extreme_values(y_nonan_nofv)
                        y_nonan_nofv_nE = y_nonan_nofv[Ev_ind]
                        x_nonan_nofv_nE = x_nonan_nofv[Ev_ind]

                        # reject values outside global ranges:
                        global_min, global_max = cf.get_global_ranges(r, sv)
                        if global_min is not None and global_max is not None:
                            gr_ind = cf.reject_global_ranges(y_nonan_nofv_nE, global_min, global_max)
                            y_nonan_nofv_nE_nogr = y_nonan_nofv_nE[gr_ind]
                            x_nonan_nofv_nE_nogr = x_nonan_nofv_nE[gr_ind]
                        else:
                            y_nonan_nofv_nE_nogr = y_nonan_nofv_nE
                            x_nonan_nofv_nE_nogr = x_nonan_nofv_nE

                        title = ' '.join((r, ms.split('-')[0]))

                        if len(y_nonan_nofv) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((r, sv))
                            else:
                                sname = '-'.join((r, m, sv))

                            # Plot all data
                            fig, ax = pf.plot_timeseries_all(x_nonan_nofv, y_nonan_nofv, sv, sv_units, stdev=None)
                            ax.set_title((title + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                         fontsize=8)
                            for etimes in end_times:
                                ax.axvline(x=etimes,  color='b', linestyle='--', linewidth=.6)

                            if global_min is not None and global_max is not None:
                                ax.axhline(y=global_min, color='r', linestyle='--', linewidth=.6)
                                ax.axhline(y=global_max, color='r', linestyle='--', linewidth=.6)

                            pf.save_fig(save_dir, sname)

                            # Plot data with extreme values, data outside global ranges and outliers removed
                            fig, ax = pf.plot_timeseries_all(x_nonan_nofv_nE_nogr, y_nonan_nofv_nE_nogr, sv, sv_units, stdev=5)
                            ax.set_title((title + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                                         fontsize=8)
                            for etimes in end_times:
                                ax.axvline(x=etimes,  color='b', linestyle='--', linewidth=.6)

                            if global_min is not None and global_max is not None:
                                ax.axhline(y=global_min, color='r', linestyle='--', linewidth=.6)
                                ax.axhline(y=global_max, color='r', linestyle='--', linewidth=.6)

                            sfile = '_'.join((sname, 'rmoutliers'))
                            pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T145838-CE02SHBP-LJ01D-10-PHSEND103-streamed-phsen_data_record/catalog.html']

        # [
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160622-CE06ISSM-RID16-06-PHSEND000-recovered_host-phsen_abcdef_dcl_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160645-CE06ISSM-RID16-06-PHSEND000-recovered_inst-phsen_abcdef_instrument/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160700-CE06ISSM-RID16-06-PHSEND000-telemetered-phsen_abcdef_dcl_instrument/catalog.html'
        #         ]


        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T145838-CE02SHBP-LJ01D-10-PHSEND103-streamed-phsen_data_record/catalog.html']

    #     [
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163408-CE06ISSM-RID16-03-DOSTAD000-recovered_host-dosta_abcdjm_ctdbp_dcl_instrument_recovered/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163419-CE06ISSM-RID16-03-DOSTAD000-recovered_inst-dosta_abcdjm_ctdbp_instrument_recovered/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163558-CE06ISSM-RID16-03-DOSTAD000-telemetered-dosta_abcdjm_ctdbp_dcl_instrument/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163751-CE09OSPM-WFP01-02-DOFSTK000-recovered_wfp-dofst_k_wfp_instrument_recovered/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163824-CE09OSPM-WFP01-02-DOFSTK000-telemetered-dofst_k_wfp_instrument/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163845-CE09OSSM-RID27-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html',
    #     'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163907-CE09OSSM-RID27-04-DOSTAD000-telemetered-dosta_abcdjm_dcl_instrument/catalog.html'
    # ]
    main(sDir, url_list)
