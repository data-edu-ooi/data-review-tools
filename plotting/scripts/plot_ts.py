#!/usr/bin/env python
"""
Created on Jan 28 2019

@author: Lori Garzio
@brief: This script is used create temperature-salinity plots, colored by time
"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import itertools
import gsw
import matplotlib.cm as cm
import functions.common as cf
import functions.plotting as pf


def return_var(dataset, raw_vars, fstring, longname):
    lst = [i for i in raw_vars if fstring in i]
    if len(lst) == 1:
        var = lst[0]
    else:
        vars = []
        for v in lst:
            try:
                ln = dataset[v].long_name
                if ln == longname:
                    vars.append(v)
            except AttributeError:
                continue

        if len(vars) > 1:
            print('More than 1 {} variable found in the file'.format(longname))
        elif len(vars) == 1:
            var = str(vars[0])
    return var


def main(sDir, url_list, start_time, end_time, preferred_only):
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
        if preferred_only == 'yes':
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
        else:
            fdatasets = datasets

        for fd in fdatasets:
            with xr.open_dataset(fd, mask_and_scale=False) as ds:
                ds = ds.swap_dims({'obs': 'time'})

                if start_time is not None and end_time is not None:
                    ds = ds.sel(time=slice(start_time, end_time))
                    if len(ds['time'].values) == 0:
                        print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                        continue

                fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)
                print('\nPlotting {} {}'.format(r, deployment))
                array = subsite[0:2]
                save_dir = os.path.join(sDir, array, subsite, refdes, 'ts_plots')
                cf.create_dir(save_dir)

                tme = ds['time'].values
                t0 = pd.to_datetime(tme.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(tme.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, refdes, method))
                filename = '-'.join(('_'.join(fname.split('_')[:-1]), 'ts', t0[:10]))

                ds_vars = list(ds.data_vars.keys())
                raw_vars = cf.return_raw_vars(ds_vars)

                xvar = return_var(ds, raw_vars, 'salinity', 'Practical Salinity')
                sal = ds[xvar].values
                sal_fv = ds[xvar]._FillValue

                yvar = return_var(ds, raw_vars, 'temp', 'Seawater Temperature')
                temp = ds[yvar].values
                temp_fv = ds[yvar]._FillValue

                press = pf.pressure_var(ds, list(ds.coords.keys()))
                if press is None:
                    press = pf.pressure_var(ds, list(ds.data_vars.keys()))
                p = ds[press].values

                # get rid of nans, 0.0s, fill values
                sind1 = (~np.isnan(sal)) & (sal != 0.0) & (sal != sal_fv)
                sal = sal[sind1]
                temp = temp[sind1]
                tme = tme[sind1]
                p = p[sind1]
                tind1 = (~np.isnan(temp)) & (temp != 0.0) & (temp != temp_fv)
                sal = sal[tind1]
                temp = temp[tind1]
                tme = tme[tind1]
                p = p[tind1]

                # reject values outside global ranges:
                global_min, global_max = cf.get_global_ranges(r, xvar)
                if any(e is None for e in [global_min, global_max]):
                    sal = sal
                    temp = temp
                    tme = tme
                    p = p
                else:
                    sgr_ind = cf.reject_global_ranges(sal, global_min, global_max)
                    sal = sal[sgr_ind]
                    temp = temp[sgr_ind]
                    tme = tme[sgr_ind]
                    p = p[sgr_ind]

                global_min, global_max = cf.get_global_ranges(r, yvar)
                if any(e is None for e in [global_min, global_max]):
                    sal = sal
                    temp = temp
                    tme = tme
                    p = p
                else:
                    tgr_ind = cf.reject_global_ranges(temp, global_min, global_max)
                    sal = sal[tgr_ind]
                    temp = temp[tgr_ind]
                    tme = tme[tgr_ind]
                    p = p[tgr_ind]

                # get rid of outliers
                soind = cf.reject_outliers(sal, 5)
                sal = sal[soind]
                temp = temp[soind]
                tme = tme[soind]
                p = p[soind]

                toind = cf.reject_outliers(temp, 5)
                sal = sal[toind]
                temp = temp[toind]
                tme = tme[toind]
                p = p[toind]

                if len(sal) > 0:  # if there are any data to plot

                    colors = cm.rainbow(np.linspace(0, 1, len(tme)))

                    # Figure out boundaries (mins and maxes)
                    #smin = sal.min() - (0.01 * sal.min())
                    #smax = sal.max() + (0.01 * sal.max())
                    if sal.max() - sal.min() < 0.2:
                        smin = sal.min() - (0.0005 * sal.min())
                        smax = sal.max() + (0.0005 * sal.max())
                    else:
                        smin = sal.min() - (0.001 * sal.min())
                        smax = sal.max() + (0.001 * sal.max())

                    if temp.max() - temp.min() <= 1:
                        tmin = temp.min() - (0.01 * temp.min())
                        tmax = temp.max() + (0.01 * temp.max())
                    elif 1 < temp.max() - temp.min() < 1.5:
                        tmin = temp.min() - (0.05 * temp.min())
                        tmax = temp.max() + (0.05 * temp.max())
                    else:
                        tmin = temp.min() - (0.1 * temp.min())
                        tmax = temp.max() + (0.1 * temp.max())

                    # Calculate how many gridcells are needed in the x and y directions and
                    # Create temp and sal vectors of appropriate dimensions
                    xdim = int(round((smax-smin)/0.1 + 1, 0))
                    if xdim == 1:
                        xdim = 2
                    si = np.linspace(0, xdim - 1, xdim) * 0.1 + smin

                    if 1.1 <= temp.max() - temp.min() < 1.7:  # if the diff between min and max temp is small
                        ydim = int(round((tmax-tmin)/0.75 + 1, 0))
                        ti = np.linspace(0, ydim - 1, ydim) * 0.75 + tmin
                    elif temp.max() - temp.min() < 1.1:
                        ydim = int(round((tmax - tmin) / 0.1 + 1, 0))
                        ti = np.linspace(0, ydim - 1, ydim) * 0.1 + tmin
                    else:
                        ydim = int(round((tmax - tmin) + 1, 0))
                        ti = np.linspace(0, ydim - 1, ydim) + tmin

                    # Create empty grid of zeros
                    mdens = np.zeros((ydim, xdim))

                    # Loop to fill in grid with densities
                    for j in range(0, ydim):
                        for i in range(0, xdim):
                            mdens[j, i] = gsw.density.rho(si[i], ti[j], np.median(p))  # calculate density using median pressure value

                    fig, ax = pf.plot_ts(si, ti, mdens, sal, temp, colors)

                    ax.set_title((title + '\n' + t0 + ' - ' + t1 + '\ncolors = time (cooler: earlier)'), fontsize=9)
                    leg_text = ('Removed {} values (SD=5)'.format(len(ds[xvar].values) - len(sal)),)
                    ax.legend(leg_text, loc='best', fontsize=6)
                    pf.save_fig(save_dir, filename)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172034-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172050-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172104-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
