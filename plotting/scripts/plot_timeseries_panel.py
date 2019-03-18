#!/usr/bin/env python
"""
Created on Oct 3 2018

@author: Lori Garzio
@brief: This script is used create timeseries panel plots of all science variables for an instrument,
deployment, and delivery method. These plots omit data outside of 5 standard deviations.  The user has the option of
selecting a specific time range to plot and only plotting data from the preferred method/stream.
"""

import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime as dt
import itertools
import functions.common as cf
import functions.plotting as pf


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

        main_sensor = r.split('-')[-1]
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        for fd in fdatasets_sel:
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
                save_dir = os.path.join(sDir, array, subsite, refdes, 'timeseries_panel_plots')
                filename = '_'.join(fname.split('_')[:-1])
                sci_vars = cf.return_science_vars(stream)

                if len(sci_vars) > 1:
                    cf.create_dir(save_dir)
                    colors = cm.jet(np.linspace(0, 1, len(sci_vars)))

                    t = ds['time'].values
                    t0 = pd.to_datetime(t.min()).strftime('%Y-%m-%dT%H:%M:%S')
                    t1 = pd.to_datetime(t.max()).strftime('%Y-%m-%dT%H:%M:%S')
                    title = ' '.join((deployment, refdes, method))

                    # Plot data with outliers removed
                    fig, ax = pf.plot_timeseries_panel(ds, t, sci_vars, colors, 5)
                    plt.xticks(fontsize=7)
                    ax[0].set_title((title + '\n' + t0 + ' - ' + t1), fontsize=7)
                    sfile = '-'.join((filename, 'timeseries_panel', t0[:10]))
                    pf.save_fig(save_dir, sfile)
                else:
                    print('Only one science variable in file, no panel plots necessary')


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172034-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172050-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172104-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']
    start_time = None  # dt.datetime(2015, 4, 20, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2016, 5, 20, 0, 0, 0)  # optional, set to None if plotting all data
    preferred_only = 'yes'  # options: 'yes', 'no'
    main(sDir, url_list, start_time, end_time, preferred_only)
