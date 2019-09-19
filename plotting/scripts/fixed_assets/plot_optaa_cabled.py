#!/usr/bin/env python
"""
Created on Sept 18 2019

@author: Lori Garzio
@brief: This script is used create plots of streamed OPTAA data near wavelength=676 nm by deployment. The
user has the option of selecting a specific time range and/or deployment to plot.
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


def main(sDir, url_list, start_time, end_time, deployment_num):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list and 'OPTAA' in rd:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        deployments = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                for ud in udatasets:  # filter out collocated data files
                    if 'OPTAA' in ud.split('/')[-1]:
                        datasets.append(ud)
                    if ud.split('/')[-1].split('_')[0] not in deployments:
                        deployments.append(ud.split('/')[-1].split('_')[0])
        deployments.sort()

        fdatasets = np.unique(datasets).tolist()
        for deploy in deployments:
            if deployment_num is not None:
                if int(deploy[-4:]) is not deployment_num:
                    print('\nskipping {}'.format(deploy))
                    continue

            rdatasets = [s for s in fdatasets if deploy in s]
            if len(rdatasets) > 0:
                sci_vars_dict = {'optical_absorption': dict(atts=dict(fv=[], units=[])),
                                 'beam_attenuation': dict(atts=dict(fv=[], units=[]))}
                for i in range(len(rdatasets)):
                #for i in range(0, 2):  ##### for testing
                    ds = xr.open_dataset(rdatasets[i], mask_and_scale=False)
                    ds = ds.swap_dims({'obs': 'time'})

                    if start_time is not None and end_time is not None:
                        ds = ds.sel(time=slice(start_time, end_time))
                        if len(ds['time'].values) == 0:
                            print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                            continue

                    if i == 0:
                        fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(rdatasets[0])
                        array = subsite[0:2]
                        filename = '_'.join(fname.split('_')[:-1])
                        save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_plots', deployment)
                        cf.create_dir(save_dir)

                    for k in sci_vars_dict.keys():
                        print('\nAppending data from {}: {}'.format(deploy, k))
                        vv = ds[k]
                        fv = vv._FillValue
                        vvunits = vv.units
                        if fv not in sci_vars_dict[k]['atts']['fv']:
                            sci_vars_dict[k]['atts']['fv'].append(fv)
                        if vvunits not in sci_vars_dict[k]['atts']['units']:
                            sci_vars_dict[k]['atts']['units'].append(vvunits)
                        if k == 'optical_absorption':
                            wavelengths = ds['wavelength_a'].values
                        elif k == 'beam_attenuation':
                            wavelengths = ds['wavelength_c'].values
                        for j in range(len(wavelengths)):
                            if (wavelengths[j] > 671.) and (wavelengths[j] < 679.):
                                wv = str(wavelengths[j])
                                try:
                                    sci_vars_dict[k][wv]
                                except KeyError:
                                    sci_vars_dict[k].update({wv: dict(values=np.array([]), time=np.array([], dtype=np.datetime64))})

                                v = vv.sel(wavelength=j).values
                                sci_vars_dict[k][wv]['values'] = np.append(sci_vars_dict[k][wv]['values'], v)
                                sci_vars_dict[k][wv]['time'] = np.append(sci_vars_dict[k][wv]['time'], ds['time'].values)

            title = ' '.join((deployment, refdes, method))

            colors = ['purple', 'green', 'orange']
            t0_array = np.array([], dtype=np.datetime64)
            t1_array = np.array([], dtype=np.datetime64)
            for var in sci_vars_dict.keys():
                print('Plotting {}'.format(var))
                plotting = []  # keep track if anything is plotted
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                [g_min, g_max] = cf.get_global_ranges(r, var)
                for idk, dk in enumerate(sci_vars_dict[var]):
                    if dk != 'atts':
                        v = sci_vars_dict[var][dk]['values']
                        n_all = len(sci_vars_dict[var][dk]['values'])
                        n_nan = np.sum(np.isnan(v))

                        # convert fill values to nans
                        v[v == sci_vars_dict[var]['atts']['fv'][0]] = np.nan
                        n_fv = np.sum(np.isnan(v)) - n_nan

                        if n_nan + n_fv < n_all:
                            # plot before global ranges are removed
                            plotting.append('yes')
                            tm = sci_vars_dict[var][dk]['time']
                            t0_array = np.append(t0_array, tm.min())
                            t1_array = np.append(t1_array, tm.max())

                            ax1.scatter(tm, v, c=colors[idk - 1], label='{} nm'.format(dk), marker='.', s=1)

                            # reject data outside of global ranges
                            if g_min is not None and g_max is not None:
                                v[v < g_min] = np.nan
                                v[v > g_max] = np.nan
                                n_grange = np.sum(np.isnan(v)) - n_fv - n_nan
                            else:
                                n_grange = 'no global ranges'

                            # plot after global ranges are removed

                            ax2.scatter(tm, v, c=colors[idk - 1], label='{} nm: rm {} GR'.format(dk, n_grange),
                                        marker='.', s=1)

                if len(plotting) > 0:
                    t0 = pd.to_datetime(t0_array.min()).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(t1_array.max()).strftime('%Y-%m-%dT%H:%M:%S')
                    ax1.grid()
                    pf.format_date_axis(ax1, fig1)
                    ax1.legend(loc='best', fontsize=7)
                    ax1.set_ylabel((var + " (" + sci_vars_dict[var]['atts']['units'][0] + ")"), fontsize=9)
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
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190904T191438036Z-CE02SHBP-LJ01D-08-OPTAAD106-streamed-optaa_sample/catalog.html']
    start_time = None  # dt.datetime(2015, 6, 3, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2019, 1, 1, 0, 0, 0)  # optional, set to None if plotting all data
    deployment_num = 2  # None or int
    main(sDir, url_list, start_time, end_time, deployment_num)
