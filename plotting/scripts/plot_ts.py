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
                filename = '-'.join(('_'.join(fname.split('_')[:-1]), 'ts'))
                save_dir = os.path.join(sDir, array, subsite, refdes, 'ts_plots')
                cf.create_dir(save_dir)

                tme = ds['time'].values
                t0 = pd.to_datetime(tme.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(tme.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, refdes, method))

                ds_vars = list(ds.data_vars.keys())
                raw_vars = cf.return_raw_vars(ds_vars)

                xvar = return_var(ds, raw_vars, 'salinity', 'Practical Salinity')
                sal = ds[xvar].values
                sal_fv = ds[xvar]._FillValue
                # get rid of nans, 0.0s, fill values
                ind = (~np.isnan(sal)) & (sal != 0.0) & (sal != sal_fv)
                sal = sal[ind]

                ind2 = cf.reject_outliers(sal, 5)
                sal = sal[ind2]

                yvar = return_var(ds, raw_vars, 'temp', 'Seawater Temperature')
                temp = ds[yvar].values
                temp = temp[ind]
                temp = temp[ind2]

                tme = tme[ind]
                tme = tme[ind2]
                colors = cm.rainbow(np.linspace(0, 1, len(tme)))

                press = pf.pressure_var(ds, list(ds.coords.keys()))
                if press is None:
                    press = pf.pressure_var(ds, list(ds.data_vars.keys()))
                p = ds[press].values[ind]
                p = p[ind2]

                # Figure out boundaries (mins and maxes)
                smin = sal.min() - (0.01 * sal.min())
                smax = sal.max() + (0.01 * sal.max())
                tmin = temp.min() - (0.01 * temp.min())
                tmax = temp.max() + (0.01 * temp.max())

                # Calculate how many gridcells are needed in the x and y directions
                xdim = int(round((smax-smin)/0.1+1, 0))
                ydim = int(round((tmax-tmin)+1, 0))

                # Create empty grid of zeros
                mdens = np.zeros((ydim, xdim))

                # Create temp and sal vectors of appropriate dimensions
                ti = np.linspace(0, ydim - 1, ydim) + tmin
                si = np.linspace(0, xdim - 1, xdim) * 0.1 + smin

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
