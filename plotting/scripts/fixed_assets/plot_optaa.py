#!/usr/bin/env python
"""
Created on June 4 2019

@author: Lori Garzio
@brief: This script is used create plots of OPTAA data near wavelength=676 nm by deployment and delivery method. The
user has the option of selecting a specific time range to plot and only plotting data from the preferred method/stream.
676 nm = proxy for chl-a
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf


def main(sDir, url_list, start_time, end_time, preferred_only):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list and 'OPTAA' in rd:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                for ud in udatasets:  # filter out collocated data files
                    if 'OPTAA' in ud.split('/')[-1]:
                        datasets.append(ud)
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

        fdatasets = np.unique(fdatasets).tolist()
        for fd in fdatasets:
            ds = xr.open_dataset(fd, mask_and_scale=False)
            ds = ds.swap_dims({'obs': 'time'})

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)
            #if deployment == 'deployment0007':
            #sci_vars = cf.return_science_vars(stream)
            sci_vars = ['optical_absorption', 'beam_attenuation']
            print('\nPlotting {} {}'.format(r, deployment))
            array = subsite[0:2]
            filename = '_'.join(fname.split('_')[:-1])
            save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', deployment)
            cf.create_dir(save_dir)

            tm = ds['time'].values
            t0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            # # add chl-a data from the collocated fluorometer
            # flor_url = [s for s in url_list if r.split('-')[0] in s and 'FLOR' in s]
            # if len(flor_url) == 1:
            #     flor_datasets = cf.get_nc_urls(flor_url)
            #     # filter out collocated datasets
            #     flor_dataset = [j for j in flor_datasets if ('FLOR' in j.split('/')[-1] and deployment in j.split('/')[-1])]
            #     if len(flor_dataset) > 0:
            #         ds_flor = xr.open_dataset(flor_dataset[0], mask_and_scale=False)
            #         ds_flor = ds_flor.swap_dims({'obs': 'time'})
            #         flor_t0 = dt.datetime.strptime(t0, '%Y-%m-%dT%H:%M:%S')
            #         flor_t1 = dt.datetime.strptime(t1, '%Y-%m-%dT%H:%M:%S')
            #         ds_flor = ds_flor.sel(time=slice(flor_t0, flor_t1))
            #         t_flor = ds_flor['time'].values
            #         flor_sci_vars = cf.return_science_vars(ds_flor.stream)
            #         for fsv in flor_sci_vars:
            #             if ds_flor[fsv].long_name == 'Chlorophyll-a Concentration':
            #                 chla = ds_flor[fsv]

            for var in sci_vars:
                print(var)
                if var == 'optical_absorption':
                    wv = ds['wavelength_a'].values
                else:
                    wv = ds['wavelength_c'].values
                vv = ds[var]
                fv = vv._FillValue
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                plotting = []  # keep track if anything is plotted
                wavelengths = []
                iwavelengths = []
                for i in range(len(wv)):
                    if (wv[i] > 671.) and (wv[i] < 679.):
                        wavelengths.append(wv[i])
                        iwavelengths.append(i)

                colors = ['purple', 'green', 'orange']
                for iw in range(len(iwavelengths)):
                    v = vv.sel(wavelength=iwavelengths[iw]).values
                    n_all = len(v)
                    n_nan = np.sum(np.isnan(v))

                    # convert fill values to nans
                    v[v == fv] = np.nan
                    n_fv = np.sum(np.isnan(v)) - n_nan

                    if n_nan + n_fv < n_all:
                        # plot before global ranges are removed
                        #fig, ax = pf.plot_optaa(tm, v, vv.name, vv.units)
                        plotting.append('yes')
                        ax1.scatter(tm, v, c=colors[iw], label='{} nm'.format(wavelengths[iw]),
                                    marker='.', s=1)

                        # reject data outside of global ranges
                        [g_min, g_max] = cf.get_global_ranges(r, var)
                        if g_min is not None and g_max is not None:
                            v[v < g_min] = np.nan
                            v[v > g_max] = np.nan
                            n_grange = np.sum(np.isnan(v)) - n_fv - n_nan
                        else:
                            n_grange = 'no global ranges'

                        # plot after global ranges are removed

                        ax2.scatter(tm, v, c=colors[iw], label='{} nm: rm {} GR'.format(wavelengths[iw], n_grange),
                                    marker='.', s=1)
                        # if iw == len(wavelengths) - 1:
                        #     ax2a = ax2.twinx()
                        #     ax2a.scatter(t_flor, chla.values, c='lime', marker='.', s=1)
                        #     ax2a.set_ylabel('Fluorometric Chl-a ({})'.format(chla.units))

                if len(plotting) > 0:
                    ax1.grid()
                    pf.format_date_axis(ax1, fig1)
                    ax1.legend(loc='best', fontsize=7)
                    ax1.set_ylabel((var + " (" + vv.units + ")"), fontsize=9)
                    ax1.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '-'.join((filename, var, t0[:10]))
                    save_file = os.path.join(save_dir, sfile)
                    fig1.savefig(str(save_file), dpi=150)

                    ax2.grid()
                    pf.format_date_axis(ax2, fig2)
                    ax2.legend(loc='best', fontsize=7)
                    ax2.set_ylabel((var + " (" + vv.units + ")"), fontsize=9)
                    title_gr = 'GR: global ranges'
                    ax2.set_title((title + '\n' + t0 + ' - ' + t1 + '\n' + title_gr), fontsize=9)
                    sfile2 = '-'.join((filename, var, t0[:10], 'rmgr'))
                    save_file2 = os.path.join(save_dir, sfile2)
                    fig2.savefig(str(save_file2), dpi=150)

                plt.close('all')


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190530T182609-CE01ISSM-RID16-01-OPTAAD000-recovered_host-optaa_dj_dcl_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190607T165934-CE01ISSM-RID16-02-FLORTD000-recovered_host-flort_sample/catalog.html']
    start_time = None  # dt.datetime(2015, 6, 3, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2019, 1, 1, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
