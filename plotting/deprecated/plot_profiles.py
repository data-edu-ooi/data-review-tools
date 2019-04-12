#!/usr/bin/env python
"""
Created on Oct 4 2018

@author: Lori Garzio
@brief: This script is used create two profile plots of raw and science variables for a mobile instrument (e.g.
profilers and gliders) by deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5
standard deviations. The user has the option of selecting a specific time range to plot.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.cm as cm
import datetime as dt
import functions.common as cf
import functions.plotting as pf


def main(sDir, f, start_time, end_time):
    ff = pd.read_csv(os.path.join(sDir, f))
    url_list = ff['outputUrl'].tolist()
    for i, u in enumerate(url_list):
        print('\nUrl {} of {}: {}'.format(i + 1, len(url_list), u))
        main_sensor = u.split('/')[-2].split('-')[4]
        datasets = cf.get_nc_urls([u])
        datasets_sel = cf.filter_collocated_instruments(main_sensor, datasets)

        for ii, d in enumerate(datasets_sel):
            print('\nDataset {} of {}: {}'.format(ii + 1, len(datasets_sel), d))
            with xr.open_dataset(d, mask_and_scale=False) as ds:
                ds = ds.swap_dims({'obs': 'time'})

                if start_time is not None and end_time is not None:
                    ds = ds.sel(time=slice(start_time, end_time))
                    if len(ds['time'].values) == 0:
                        print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                        continue

                fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)
                vars = ds.data_vars.keys()

                if 'MOAS' in subsite and 'CTD' in main_sensor:  # for glider CTDs, pressure is a coordinate
                    pressure = 'sci_water_pressure_dbar'
                else:
                    pressure = pf.pressure_var(ds, vars)

                raw_vars = cf.return_raw_vars(vars)
                raw_vars = [s for s in raw_vars if s not in [pressure]]  # remove pressure from sci_vars

                save_dir = os.path.join(sDir, subsite, refdes, 'profile_plots', deployment)
                cf.create_dir(save_dir)

                t = ds['time'].values
                t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, refdes, method))

                colors = cm.rainbow(np.linspace(0, 1, len(t)))

                y = ds[pressure]

                print('Plotting variables...')
                for var in raw_vars:
                    print(var)
                    x = ds[var]

                    # Plot all data
                    xlabel = var + " (" + x.units + ")"
                    ylabel = pressure + " (" + y.units + ")"

                    fig, ax = pf.plot_profiles(x, y, colors, ylabel, xlabel, stdev=None)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '_'.join((fname[0:-46], x.name))
                    pf.save_fig(save_dir, sfile)

                    # Plot data with outliers removed
                    fig, ax = pf.plot_profiles(x, y, colors, ylabel, xlabel, stdev=5)
                    ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                    sfile = '_'.join((fname[0:-46], x.name, 'rmoutliers'))
                    pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/leila/Documents/NSFEduSupport/review/test/'
    f = '/Users/leila/Documents/NSFEduSupport/review/request/20190218T1043/data_request_summary_20190218T1043.csv'
    start_time = None  # dt.datetime(2016, 6, 1, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2017, 10, 1, 0, 0, 0)  # optional, set to None if plotting all data
    main(sDir, f, start_time, end_time)
