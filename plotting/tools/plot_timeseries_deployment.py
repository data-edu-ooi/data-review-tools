#!/usr/bin/env python
"""
Created on Oct 2 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of each science variable for a reference designator by
deployment and delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
"""

import os
import pandas as pd
import xarray as xr
import functions.common as cf
import functions.plotting as pf


def main(sDir, f):
    ff = pd.read_csv(os.path.join(sDir, f))
    datasets = cf.get_nc_urls(ff['outputUrl'].tolist())
    for d in datasets:
        print d
        with xr.open_dataset(d, mask_and_scale=False) as ds:
            ds = ds.swap_dims({'obs': 'time'})
            fname = d.split('/')[-1].split('.nc')[0]
            subsite = ds.subsite
            node = ds.node
            sensor = ds.sensor
            refdes = '-'.join((subsite, node, sensor))
            method = ds.collection_method
            deployment = fname[0:14]
            save_dir = os.path.join(sDir, subsite, refdes, deployment)
            cf.create_dir(save_dir)

            sci_vars = pf.science_vars(ds.data_vars.keys())

            t = ds['time'].data
            t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            for var in sci_vars:
                y = ds[var]

                # Plot all data
                fig, ax = pf.plot_timeseries(t, y, stdev=None)
                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                sfile = '_'.join((fname, y.name))
                pf.save_fig(save_dir, sfile)

                # Plot data with outliers removed
                fig, ax = pf.plot_timeseries(t, y, stdev=5)
                ax.set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
                sfile = '_'.join((fname, y.name, 'rmoutliers'))
                pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary.csv'
    main(sDir, f)
