#!/usr/bin/env python
"""
Created on Oct 3 2018

@author: Lori Garzio
@brief: This script is used create timeseries panel plots of all science variables for an instrument,
deployment, and delivery method. These plots omit data outside of 5 standard deviations.
"""

import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import functions.common as cf
import functions.plotting as pf


def main(sDir, f):
    ff = pd.read_csv(os.path.join(sDir, f))
    url_list = ff['outputUrl'].tolist()
    for u in url_list:
        main_sensor = u.split('/')[-2].split('-')[4]
        datasets = cf.get_nc_urls([u])
        datasets_sel = cf.filter_collocated_instruments(main_sensor, datasets)

        for d in datasets_sel:
            print d
            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)
            save_dir = os.path.join(sDir, subsite, refdes)
            cf.create_dir(save_dir)

            sci_vars = cf.return_science_vars(stream)

            colors = cm.jet(np.linspace(0, 1, len(sci_vars)))

            with xr.open_dataset(d, mask_and_scale=False) as ds:
                ds = ds.swap_dims({'obs': 'time'})
                t = ds['time'].data
                t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, refdes, method))

                # Plot data with outliers removed
                fig, ax = pf.plot_timeseries_panel(ds, t, sci_vars, colors, 5)
                plt.xticks(fontsize=7)
                ax[0].set_title((title + '\n' + t0 + ' - ' + t1), fontsize=7)
                sfile = '_'.join((fname, 'timeseries_panel'))
                pf.save_fig(save_dir, sfile)


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    f = 'data_request_summary_copy.csv'
    main(sDir, f)
