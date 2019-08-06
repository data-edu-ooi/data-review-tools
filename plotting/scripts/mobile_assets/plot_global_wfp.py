#!/usr/bin/env python
"""
Created on Aug 5 2019 by Lori Garzio

@brief: This script is used to create profile plots for instruments on the upper and lower profilers on global
WFP. Excludes erroneous data and data outside of global ranges. Each plot contains data from one deployment and one
science variable.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import itertools
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf


def main(url_list, sDir, stime, etime):
    if len(url_list) != 2:
        print('Please provide 2 reference designators for plotting')
    else:
        uu0 = url_list[0]
        uu1 = url_list[1]
        rd0 = uu0.split('/')[-2][20:47]
        rd1 = uu1.split('/')[-2][20:47]
        array = rd0[0:2]
        inst = rd0.split('-')[-1]

        datasets0 = []
        datasets1 = []
        for i in range(len(url_list)):
            udatasets = cf.get_nc_urls([url_list[i]])
            if i == 0:
                datasets0.append(udatasets)
            else:
                datasets1.append(udatasets)

        datasets0 = list(itertools.chain(*datasets0))
        datasets1 = list(itertools.chain(*datasets1))

        main_sensor0 = rd0.split('-')[-1]
        main_sensor1 = rd1.split('-')[-1]
        fdatasets0_sel = cf.filter_collocated_instruments(main_sensor0, datasets0)
        fdatasets1_sel = cf.filter_collocated_instruments(main_sensor1, datasets1)

        deployments = [dd.split('/')[-1].split('_')[0] for dd in fdatasets0_sel]

        for d in deployments:
            fd0 = [x for x in fdatasets0_sel if d in x]
            fd1 = [x for x in fdatasets1_sel if d in x]

            ds0 = xr.open_dataset(fd0[0], mask_and_scale=False)
            ds0 = ds0.swap_dims({'obs': 'time'})
            ds1 = xr.open_dataset(fd1[0], mask_and_scale=False)
            ds1 = ds1.swap_dims({'obs': 'time'})

            if stime is not None and etime is not None:
                ds0 = ds0.sel(time=slice(stime, etime))
                ds1 = ds1.sel(time=slice(stime, etime))
                if len(ds0['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd0[0])
            sci_vars = cf.return_science_vars(stream)

            save_dir_profile = os.path.join(sDir, array, subsite, inst, 'profile_plots', deployment)
            cf.create_dir(save_dir_profile)

            # get pressure variable
            pvarname, y1, y_units, press, y_fillvalue = cf.add_pressure_to_dictionary_of_sci_vars(ds0)

            for sv in sci_vars:
                print('')
                print(sv)
                if 'pressure' not in sv:
                    fig, ax = plt.subplots()
                    plt.margins(y=.08, x=.02)
                    plt.grid()
                    title = ' '.join((deployment, subsite, inst, method))
                    sname = '-'.join((subsite, inst, method, sv))
                    for i in range(len(url_list)):
                        if i == 0:
                            ds = ds0
                        else:
                            ds = ds1
                        t = ds['time'].values
                        zpressure = ds[pvarname].values
                        z1 = ds[sv].values
                        fv = ds[sv]._FillValue
                        sv_units = ds[sv].units

                        # Check if the array is all NaNs
                        if sum(np.isnan(z1)) == len(z1):
                            print('Array of all NaNs - skipping plot.')
                            continue

                        # Check if the array is all fill values
                        elif len(z1[z1 != fv]) == 0:
                            print('Array of all fill values - skipping plot.')
                            continue

                        else:
                            # get rid of 0.0 data
                            if sv == 'salinity':
                                ind = z1 > 1
                            elif sv == 'density':
                                ind = z1 > 1000
                            elif sv == 'conductivity':
                                ind = z1 > 0.1
                            elif sv == 'dissolved_oxygen':
                                ind = z1 > 160
                            elif sv == 'estimated_oxygen_concentration':
                                ind = z1 > 200
                            else:
                                ind = z1 > 0
                            # if sv == 'sci_flbbcd_chlor_units':
                            #     ind = ndata < 7.5
                            # elif sv == 'sci_flbbcd_cdom_units':
                            #     ind = ndata < 25
                            # else:
                            #     ind = ndata > 0.0

                            # if 'CTD' in r:
                            #     ind = zpressure > 0.0
                            # else:
                            #     ind = ndata > 0.0

                            lenzero = np.sum(~ind)
                            dtime = t[ind]
                            zpressure = zpressure[ind]
                            zdata = z1[ind]

                            if len(dtime) > 0:
                                ax.scatter(zdata, zpressure, s=2, edgecolor='None')

                    xlabel = sv + " (" + sv_units + ")"
                    ylabel = press[0] + " (" + y_units[0] + ")"

                    ax.invert_yaxis()
                    # plt.xlim([-0.5, 0.5])
                    ax.set_xlabel(xlabel, fontsize=9)
                    ax.set_ylabel(ylabel, fontsize=9)
                    ax.set_title(title + '\nWFP02 (blue) & WFP03 (orange)', fontsize=9)
                    fig.tight_layout()
                    pf.save_fig(save_dir_profile, sname)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    start_time = None  # dt.datetime(2016, 10, 29, 0, 0, 0)  # optional, set to None if plotting all data
    end_time = None  # dt.datetime(2018, 1, 12, 0, 0, 0)  # optional, set to None if plotting all data
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = [
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190802T150238833Z-GP02HYPM-WFP02-04-CTDPFL000-recovered_wfp-ctdpf_ckl_wfp_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190802T150239880Z-GP02HYPM-WFP03-04-CTDPFL000-recovered_wfp-ctdpf_ckl_wfp_instrument_recovered/catalog.html']

    main(url_list, sDir, start_time, end_time)
