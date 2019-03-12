#!/usr/bin/env python
"""
Created on Nov 8 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of telemetered and recovered science variables for a reference
designator where the Long Names of the two variables are the same: 1) plot all data, 2) plot data, omitting outliers
beyond 5 standard deviations. The user has the option of selecting a specific time range to plot.
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import functions.common as cf
import functions.plotting as pf


def compare_plot_datasets(df, r, start_time, end_time, sDir):
    names = df.columns
    for d, row in df.iterrows():
        print('\n{}'.format(d))
        for i, n in enumerate(names):
            ii = i + 1
            if ii > 1:
                f1 = row[n]
                try:
                    if np.isnan(f1) is True:
                        continue
                except TypeError:
                    for x in range(ii - 1):
                        f0 = row[names[x]]
                        try:
                            if np.isnan(f0) is True:
                                continue
                        except TypeError:
                            compare = '{} {}'.format(names[x], n)

                            if len(f0) == 1:
                                ds0 = xr.open_dataset(f0[0])
                                ds0 = ds0.swap_dims({'obs': 'time'})
                            else:
                                ds0 = xr.open_mfdataset(f0)
                                ds0 = ds0.swap_dims({'obs': 'time'})
                                ds0 = ds0.chunk({'time': 100})
                            splt0 = compare.split(' ')[0].split('-')
                            ds0_sci_vars = cf.return_science_vars(splt0[1])
                            ds0_method = splt0[0]

                            if start_time is not None and end_time is not None:
                                ds0 = ds0.sel(time=slice(start_time, end_time))

                                if len(ds0['time'].values) == 0:
                                    print('No {} data to plot for specified time range: ({} to {})'.format(ds0_method,
                                                                                                 start_time, end_time))
                                    continue

                            if len(f1) == 1:
                                ds1 = xr.open_dataset(f1[0])
                                ds1 = ds1.swap_dims({'obs': 'time'})
                            else:
                                ds1 = xr.open_mfdataset(f1)
                                ds1 = ds1.swap_dims({'obs': 'time'})
                                ds1 = ds1.chunk({'time': 100})
                            splt1 = compare.split(' ')[1].split('-')
                            ds1_sci_vars = cf.return_science_vars(splt1[1])
                            ds1_method = splt1[0]

                            if start_time is not None and end_time is not None:
                                ds1 = ds1.sel(time=slice(start_time, end_time))
                                if len(ds1['time'].values) == 0:
                                    print ('No {} data to plot for specified time range: ({} to {})'.format(ds1_method,
                                                                                                           start_time,
                                                                                                           end_time))
                                    continue

                            t0 = ds0['time']
                            t1 = ds1['time']

                            # find where the variable long names are the same
                            ds0names = long_names(ds0, ds0_sci_vars)
                            ds0names.rename(columns={'name': 'name_ds0'}, inplace=True)
                            ds1names = long_names(ds1, ds1_sci_vars)
                            ds1names.rename(columns={'name': 'name_ds1'}, inplace=True)
                            mapping = pd.merge(ds0names, ds1names, on='long_name', how='inner')
                            print('----------------------')
                            print('{}: {}'.format(d, compare))
                            print('----------------------')

                            subsite = r.split('-')[0]
                            array = subsite[0:2]
                            if start_time is not None and end_time is not None:
                                stime = start_time.strftime('%Y-%m-%d')
                                etime = end_time.strftime('%Y-%m-%d')
                                ext = '-'.join((d, compare)) + '-' + stime + 'to' + etime #.join((ds0_method, ds1_method
                                save_dir = os.path.join(sDir, array, subsite, r, 'method_compare_plots', ext)
                            else:
                                save_dir = os.path.join(sDir, array, subsite, r, 'method_compare_plots',
                                                        '-'.join((ds0_method, ds1_method)))
                            cf.create_dir(save_dir)

                            for rr in mapping.itertuples():
                                index, name_ds0, long_name, name_ds1 = rr
                                print(long_name)

                                ds0_var = ds0[name_ds0]
                                ds1_var = ds1[name_ds1]

                                # reject NaNs
                                nan0_ind = ~np.isnan(ds0_var.data)
                                ds0_nonan = ds0_var.data[nan0_ind]

                                nan1_ind = ~np.isnan(ds1_var.data)
                                ds1_nonan = ds1_var.data[nan1_ind]

                                # only plot if both arrays have data
                                if len(ds0_nonan) > 0 and len(ds1_nonan) > 0:
                                    # Plot all data
                                    fig, ax = pf.plot_timeseries_compare(t0, t1, ds0_var, ds1_var, ds0_method, ds1_method,
                                                                         long_name, stdev=None)

                                    title = ' '.join((d, r, '{} vs {}'.format(ds0_method, ds1_method)))
                                    ax.set_title(title, fontsize=9)
                                    sfile = '_'.join((d, r, long_name))
                                    pf.save_fig(save_dir, sfile)

                                    # Plot data with outliers removed
                                    fig, ax = pf.plot_timeseries_compare(t0, t1, ds0_var, ds1_var, ds0_method, ds1_method,
                                                                         long_name, stdev=5)

                                    title = ' '.join((d, r, '{} vs {}'.format(ds0_method, ds1_method)))
                                    ax.set_title(title, fontsize=9)
                                    sfile = '_'.join((d, r, long_name, 'rmoutliers'))
                                    pf.save_fig(save_dir, sfile)


def long_names(dataset, vars):
    name = []
    long_name = []
    for v in vars:
        name.append(v)  # list of recovered variable names

        try:
            longname = dataset[v].long_name
        except AttributeError:
            longname = vars

        long_name.append(longname)

    return pd.DataFrame({'name': name, 'long_name': long_name})


def main(sDir, url_list, start_time, end_time):
    # get summary lists of reference designators and delivery methods
    rd_list = []
    rdm_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        rdm = '-'.join((rd, elements[5]))
        if rd not in rd_list:
            rd_list.append(rd)
        if rdm not in rdm_list:
            rdm_list.append(rdm)

    for r in rd_list:
        rdm_filtered = [k for k in rdm_list if r in k]
        dinfo = {}
        if len(rdm_filtered) == 1:
            print('Only one delivery method provided - no comparison.')
            continue

        elif len(rdm_filtered) > 1 & len(rdm_filtered) <= 3:
            print('\nComparing data from different methods for: {}'.format(r))
            for i in range(len(rdm_filtered)):
                urls = [x for x in url_list if rdm_filtered[i] in x]
                for u in urls:
                    splitter = u.split('/')[-2].split('-')
                    catalog_rms = '-'.join((r, splitter[-2], splitter[-1]))
                    udatasets = cf.get_nc_urls([u])
                    deployments = [str(k.split('/')[-1][0:14]) for k in udatasets]
                    udeploy = np.unique(deployments).tolist()
                    for ud in udeploy:
                        rdatasets = [s for s in udatasets if ud in s]
                        datasets = []
                        for dss in rdatasets:  # filter out collocated data files
                            if catalog_rms == dss.split('/')[-1].split('_20')[0][15:]:
                                datasets.append(dss)
                        if len(datasets) == 0:
                            print('no data for ', ud)
                        else:
                            file_ms_lst = []
                            for dataset in datasets:
                                splt = dataset.split('/')[-1].split('_20')[0].split('-')
                                file_ms_lst.append('-'.join((splt[-2], splt[-1])))
                            file_ms = np.unique(file_ms_lst).tolist()[0]
                            try:
                                dinfo[file_ms]
                            except KeyError:
                                dinfo[file_ms] = {}
                            dinfo[file_ms].update({ud: datasets})

        else:
            print('More than 3 methods provided. Please provide fewer datasets for analysis.')
            continue

        dinfo_df = pd.DataFrame(dinfo)

        umethods = []
        ustreams = []
        for k in dinfo.keys():
            umethods.append(k.split('-')[0])
            ustreams.append(k.split('-')[1])

        if len(np.unique(ustreams)) > len(np.unique(umethods)):  # if there is more than 1 stream per delivery method
            method_stream_df = cf.stream_word_check(dinfo)
            for cs in (np.unique(method_stream_df['stream_name_compare'])).tolist():
                print('Common stream_name: {}'.format(cs))
                method_stream_list = []
                for row in method_stream_df.itertuples():
                    index, method, stream_name, stream_name_compare = row
                    if stream_name_compare == cs:
                        method_stream_list.append('-'.join((method, stream_name)))
                dinfo_df_filtered = dinfo_df[method_stream_list]
                compare_plot_datasets(dinfo_df_filtered, r, start_time, end_time, sDir)

        else:
            compare_plot_datasets(dinfo_df, r, start_time, end_time, sDir)


if __name__ == '__main__':
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235715-CP03ISSM-MFD37-04-DOSTAD000-telemetered-dosta_abcdjm_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235659-CP03ISSM-MFD37-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html']

    start_time = dt.datetime(2017, 10, 31, 0, 0, 0)  # None  # optional, set to None if plotting all data
    end_time =  dt.datetime(2017, 11, 4, 0, 0, 0)   # None # optional, set to None if plotting all data

    main(sDir, url_list, start_time, end_time)

    # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235321-CP03ISSM-MFD37-03-CTDBPD000-telemetered-ctdbp_cdef_dcl_instrument/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235146-CP03ISSM-MFD37-03-CTDBPD000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235133-CP03ISSM-MFD37-03-CTDBPD000-recovered_host-ctdbp_cdef_dcl_instrument_recovered/catalog.html']

        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235528-CP03ISSM-RID27-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235509-CP03ISSM-RID27-03-CTDBPC000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181212T235337-CP03ISSM-RID27-03-CTDBPC000-recovered_host-ctdbp_cdef_dcl_instrument_recovered/catalog.html']

        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000821-CP03ISSM-SBD11-06-METBKA000-telemetered-metbk_a_dcl_instrument/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000115-CP03ISSM-SBD11-06-METBKA000-recovered_host-metbk_a_dcl_instrument_recovered/catalog.html']

        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000154-CP03ISSM-SBD11-06-METBKA000-telemetered-metbk_hourly/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000135-CP03ISSM-SBD11-06-METBKA000-recovered_host-metbk_hourly/catalog.html']

        #


        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T020745-GS01SUMO-RII11-02-PHSENE041-telemetered-phsen_abcdef_imodem_instrument/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T020724-GS01SUMO-RII11-02-PHSENE041-recovered_host-phsen_abcdef_imodem_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T020706-GS01SUMO-RII11-02-PHSENE041-recovered_inst-phsen_abcdef_instrument/catalog.html']


        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T003335-GA01SUMO-RII11-02-PHSENE042-telemetered-phsen_abcdef_imodem_instrument/catalog.html',
        #  'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T003322-GA01SUMO-RII11-02-PHSENE042-recovered_host-phsen_abcdef_imodem_instrument_recovered/catalog.html',
        #  'httpss://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190112T003307-GA01SUMO-RII11-02-PHSENE042-recovered_inst-phsen_abcdef_instrument/catalog.html']

        # ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123336-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123345-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/ooidatateam@gmail.com/20181026T123354-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']

