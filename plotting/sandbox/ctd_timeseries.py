#!/usr/bin/env python
"""
Created on Oct 3 2018

@author: Lori Garzio
@brief: This script is used create a timeseries plot of CTD data containing all science variables by instrument,
deployment, and delivery method. These plots omit data outside of 5 standard deviations.
"""

import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import functions.common as cf
import functions.plotting as pf


def main(sDir, f):
    ff = pd.read_csv(os.path.join(sDir, f))
    datasets = cf.get_nc_urls(ff['outputUrl'].tolist())
    for d in datasets:
        print(d)
        fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)
        save_dir = os.path.join(sDir, subsite, refdes, deployment)
        cf.create_dir(save_dir)
        
        sci_vars = cf.return_science_vars(stream)

        colors = cm.jet(np.linspace(0, 1, len(sci_vars)))

        with xr.open_dataset(d, mask_and_scale=False) as ds:
            ds = ds.swap_dims({'obs': 'time'})
            t = ds['time'].data
            t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
            t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
            title = ' '.join((deployment, refdes, method))

            fig, ax = plt.subplots()
            axes = [ax]
            for i in range(len(sci_vars)):
                if i > 0:
                    axes.append(ax.twinx())  # twin the x-axis to make independent y-axes

            fig.subplots_adjust(right=0.6)
            right_additive = (0.98-0.6)/float(5)

            for i in range(len(sci_vars)):
                if i > 0:
                    axes[i].spines['right'].set_position(('axes', 1. + right_additive * i))
                y = ds[sci_vars[i]]

                ind = cf.reject_outliers(y, 5)
                yD = y.data[ind]
                x = t[ind]

                #yD = y.data
                c = colors[i]
                axes[i].plot(x, yD, '.', markersize=2, color=c)
                axes[i].set_ylabel((y.name + " (" + y.units + ")"), color=c, fontsize=9)
                axes[i].tick_params(axis='y', colors=c)
                if i == len(sci_vars) - 1:  # if the last variable has been plotted
                    pf.format_date_axis(axes[i], fig)

            axes[0].set_title((title + '\n' + t0 + ' - ' + t1), fontsize=9)
            sfile = '_'.join((fname, 'timeseries'))
            pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary.csv'
    main(sDir, f)
