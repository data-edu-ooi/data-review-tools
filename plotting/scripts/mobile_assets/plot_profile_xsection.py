#!/usr/bin/env python
"""
Created on Feb 2019 by Leila Belabbassi
Modified on Apr 17 2019 by Lori Garzio

@brief: This script is used to create initial profile plots and 3D color scatter plots for instruments on mobile
platforms (WFP & Gliders). Also produces 4D color scatter plots for gliders only. Excludes erroneous data and data
outside of global ranges. Each plot contains data from one deployment and one science variable.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
import itertools
from mpl_toolkits.mplot3d import Axes3D  # need this for 4D scatter plot
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
import functions.group_by_timerange as gt


def main(url_list, sDir, deployment_num, start_time, end_time, preferred_only, n_std, inpercentile, zcell_size):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list and 'ENG' not in rd:
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
                    try:
                        rms = '-'.join((r, row[ii]))
                    except TypeError:
                        continue
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
            part_d = fd.split('/')[-1]
            print('\n{}'.format(part_d))
            ds = xr.open_dataset(fd, mask_and_scale=False)
            ds = ds.swap_dims({'obs': 'time'})

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(fd)
            array = subsite[0:2]
            sci_vars = cf.return_science_vars(stream)

            if 'CE05MOAS' in r or 'CP05MOAS' in r:  # for coastal gliders, get m_water_depth for bathymetry
                eng = '-'.join((r.split('-')[0], r.split('-')[1], '00-ENG000000', method, 'glider_eng'))
                eng_url = [s for s in url_list if eng in s]
                if len(eng_url) == 1:
                    eng_datasets = cf.get_nc_urls(eng_url)
                    # filter out collocated datasets
                    eng_dataset = [j for j in eng_datasets if (eng in j.split('/')[-1] and deployment in j.split('/')[-1])]
                    if len(eng_dataset) > 0:
                        ds_eng = xr.open_dataset(eng_dataset[0], mask_and_scale=False)
                        t_eng = ds_eng['time'].values
                        m_water_depth = ds_eng['m_water_depth'].values

                        # m_altimeter_status = 0 means a good reading (not nan or -1)
                        eng_ind = ds_eng['m_altimeter_status'].values == 0
                        m_water_depth = m_water_depth[eng_ind]
                        t_eng = t_eng[eng_ind]
                    else:
                        print('No engineering file for deployment {}'.format(deployment))
                        m_water_depth = None
                        t_eng = None
                else:
                    m_water_depth = None
                    t_eng = None
            else:
                m_water_depth = None
                t_eng = None

            if deployment_num is not None:
                if int(deployment.split('0')[-1]) is not deployment_num:
                    print(type(int(deployment.split('0')[-1])), type(deployment_num))
                    continue

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue
                stime = start_time.strftime('%Y-%m-%d')
                etime = end_time.strftime('%Y-%m-%d')
                ext = stime + 'to' + etime  # .join((ds0_method, ds1_method
                save_dir_profile = os.path.join(sDir, array, subsite, refdes, 'profile_plots', deployment, ext)
                save_dir_xsection = os.path.join(sDir, array, subsite, refdes, 'xsection_plots', deployment, ext)
                save_dir_4d = os.path.join(sDir, array, subsite, refdes, 'xsection_plots_4d', deployment, ext)
            else:
                save_dir_profile = os.path.join(sDir, array, subsite, refdes, 'profile_plots', deployment)
                save_dir_xsection = os.path.join(sDir, array, subsite, refdes, 'xsection_plots', deployment)
                save_dir_4d = os.path.join(sDir, array, subsite, refdes, 'xsection_plots_4d', deployment)

            tm = ds['time'].values
            try:
                ds_lat = ds['lat'].values
            except KeyError:
                ds_lat = None
                print('No latitude variable in file')
            try:
                ds_lon = ds['lon'].values
            except KeyError:
                ds_lon = None
                print('No longitude variable in file')

            # get pressure variable
            y, y_units, press = cf.add_pressure_to_dictionary_of_sci_vars(ds)

            for sv in sci_vars:
                print(sv)
                if 'pressure' not in sv:
                    z = ds[sv].values
                    fv = ds[sv]._FillValue
                    sv_units = ds[sv].units

                    # Check if the array is all NaNs
                    if sum(np.isnan(z)) == len(z):
                        print('Array of all NaNs - skipping plot.')
                        continue

                    # Check if the array is all fill values
                    elif len(z[z != fv]) == 0:
                        print('Array of all fill values - skipping plot.')
                        continue

                    else:
                        # reject erroneous data
                        dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max, lat, lon = \
                            cf.reject_erroneous_data(r, sv, tm, y, z, fv, ds_lat, ds_lon)

                        # get rid of 0.0 data
                        if 'CTD' in r:
                            ind = zpressure > 0.0
                        else:
                            ind = ndata > 0.0

                        lenzero = np.sum(~ind)
                        dtime = dtime[ind]
                        zpressure = zpressure[ind]
                        ndata = ndata[ind]

                        t0 = pd.to_datetime(dtime.min()).strftime('%Y-%m-%dT%H:%M:%S')
                        t1 = pd.to_datetime(dtime.max()).strftime('%Y-%m-%dT%H:%M:%S')
                        title = ' '.join((deployment, refdes, method)) + '\n' + t0 + ' to ' + t1

                        # create data groups
                        columns = ['tsec', 'dbar', str(sv)]
                        min_r = int(round(min(zpressure) - zcell_size))
                        max_r = int(round(max(zpressure) + zcell_size))
                        ranges = list(range(min_r, max_r, zcell_size))

                        groups, d_groups = gt.group_by_depth_range(dtime, zpressure, ndata, columns, ranges)

                        if 'scatter' in sv:
                            n_std = None  # to use percentile
                        else:
                            n_std = n_std

                        #  get percentile analysis for printing on the profile plot
                        y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex = cf.reject_timestamps_in_groups(
                            groups, d_groups, n_std, inpercentile)

                        """
                        Plot data
                        """
                        if len(dtime) > 0:
                            cf.create_dir(save_dir_profile)
                            cf.create_dir(save_dir_xsection)
                            sname = '-'.join((r, method, sv))
                            sfile = '_'.join(('rm_erroneous_data', sname))

                            '''
                            profile plot
                            '''
                            xlabel = sv + " (" + sv_units + ")"
                            ylabel = press[0] + " (" + y_units[0] + ")"
                            clabel = 'Time'

                            fig, ax = pf.plot_profiles(ndata, zpressure, dtime, ylabel, xlabel, clabel, stdev=None)

                            ax.set_title(title, fontsize=9)
                            ax.plot(n_avg, y_avg, '-k')
                            ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                            leg_text = (
                                'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                                '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                                '\n(black) data average in {} dbar segments'.format(zcell_size) +
                                '\n(magenta) upper and lower {} percentile envelope in {} dbar segments'.format(
                                    inpercentile, zcell_size),)
                            ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            fig.tight_layout()
                            pf.save_fig(save_dir_profile, sfile)

                            '''
                            xsection plot
                            '''
                            clabel = sv + " (" + sv_units + ")"
                            ylabel = press[0] + " (" + y_units[0] + ")"

                            # plot non-erroneous data
                            fig, ax, bar = pf.plot_xsection(subsite, dtime, zpressure, ndata, clabel, ylabel, t_eng,
                                                            m_water_depth, inpercentile=None, stdev=None)

                            ax.set_title(title, fontsize=9)
                            leg_text = (
                                'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                                '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero),
                            )
                            ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                            fig.tight_layout()
                            pf.save_fig(save_dir_xsection, sfile)

                            '''
                            4D plot for gliders only
                            '''
                            if 'MOAS' in r:
                                if ds_lat is not None and ds_lon is not None:
                                    cf.create_dir(save_dir_4d)
                                    lat = lat[ind]
                                    lon = lon[ind]

                                    clabel = sv + " (" + sv_units + ")"
                                    zlabel = press[0] + " (" + y_units[0] + ")"

                                    fig = plt.figure()
                                    ax = fig.add_subplot(111, projection='3d')
                                    sct = ax.scatter(lon, lat, zpressure, c=ndata, s=2)
                                    cbar = plt.colorbar(sct, label=clabel, extend='both')
                                    cbar.ax.tick_params(labelsize=8)
                                    ax.invert_zaxis()
                                    ax.view_init(25, 32)
                                    ax.invert_xaxis()
                                    ax.invert_yaxis()
                                    ax.set_zlabel(zlabel, fontsize=9)
                                    ax.set_ylabel('Latitude', fontsize=9)
                                    ax.set_xlabel('Longitude', fontsize=9)

                                    ax.set_title(title, fontsize=9)
                                    pf.save_fig(save_dir_4d, sfile)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

    """
    define time range: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    """
    start_time = None  # dt.datetime(2014, 12, 1)
    end_time = None  # dt.datetime(2015, 5, 2)

    '''
    define filters standard deviation, percentile
    '''
    n_std = None
    inpercentile = 5

    '''
    define the depth cell_size for data grouping 
    '''
    zcell_size = 10

    ''''
    define deployment number and indicate if only the preferred data should be plotted
    '''
    deployment_num = 1
    preferred_only = 'yes'  # options: 'yes', 'no'

    '''
    output directory, and data files URL location
    '''
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html']
    #url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021350-CE09OSPM-WFP01-04-FLORTK000-telemetered-flort_sample/catalog.html']

    '''
    call in main function with the above attributes
    '''
    main(url_list, sDir, deployment_num, start_time, end_time, preferred_only, n_std, inpercentile, zcell_size)
