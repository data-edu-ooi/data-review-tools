#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for all deployments of a reference
designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 3 standard deviations.
"""

import os
import pandas as pd
import xarray as xr
import functions.common as cf
import functions.plotting as pf


def main(sDir, f):
    ff = pd.read_csv(os.path.join(sDir, f))
    url_list = ff['outputUrl'].tolist()
    for u in url_list:
        main_sensor = u.split('/')[-2].split('-')[4]
        datasets = cf.get_nc_urls([u])

        datasets_sel = cf.filter_collocated_instruments(main_sensor, datasets)

        deployments = []
        end_times = []
        for d in datasets_sel:
            deployments.append(int(d.split('/')[-1][13]))
            end_times.append(pd.to_datetime(d.split('/')[-1][-18:-3]))

        # get global attributes from one file
        fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(datasets_sel[0])
        save_dir = os.path.join(sDir, subsite, refdes, 'timeseries_plots_all', method)
        cf.create_dir(save_dir)
        sname = '-'.join((refdes, method, stream))

        with xr.open_mfdataset(datasets_sel, mask_and_scale=False) as ds:
            raw_vars = cf.return_raw_vars(ds.data_vars.keys())
            ds = ds.swap_dims({'obs': 'time'})
            t = ds['time'].data
            t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((refdes, method))

            for var in raw_vars:
                print(var)
                y = ds[var]

                # Plot all data
                fig, ax = pf.plot_timeseries(t, y, stdev=None)
                ax.set_title((title + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                             fontsize=8)
                for etimes in end_times:
                    ax.axvline(x=etimes,  color='k', linestyle='--', linewidth=.6)
                sfile = '_'.join((sname, y.name))
                pf.save_fig(save_dir, sfile)

                # Plot data with outliers removed
                fig, ax = pf.plot_timeseries(t, y, stdev=3)
                ax.set_title((title + '\nDeployments: ' + str(sorted(deployments)) + '\n' + t0 + ' - ' + t1),
                             fontsize=8)
                for etimes in end_times:
                    ax.axvline(x=etimes,  color='k', linestyle='--', linewidth=.6)
                sfile = '_'.join((sname, y.name, 'rmoutliers'))
                pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary.csv'
    main(sDir, f)
