#!/usr/bin/env python
"""
Created on Mar 12 2019 by Leila Belabbassi
@brief plot glider tracks
"""
import os
import requests
import matplotlib
import cartopy.crs as ccrs
import functions.plotting as pf
import functions.common as cf
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle



def main(url_list, sDir, plot_type, start_time, end_time, deployment_num):
    for i, u in enumerate(url_list):
        elements = u.split('/')[-2].split('-')
        r = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = u.split(r + '-')[1].split('/')[0]
        subsite = r.split('-')[0]
        array = subsite[0:2]
        main_sensor = r.split('-')[-1]
        datasets = cf.get_nc_urls([u])
        datasets_sel = cf.filter_collocated_instruments(main_sensor, datasets)

        save_dir = os.path.join(sDir, array, subsite, r, plot_type)
        cf.create_dir(save_dir)
        sname = '-'.join((r, ms, 'track'))

        print('Appending....')
        sh = pd.DataFrame()
        deployments = []
        end_times = []
        for ii, d in enumerate(datasets_sel):
            print('\nDataset {} of {}: {}'.format(ii + 1, len(datasets_sel), d.split('/')[-1]))
            ds = xr.open_dataset(d, mask_and_scale=False)
            ds = ds.swap_dims({'obs': 'time'})

            if start_time is not None and end_time is not None:
                ds = ds.sel(time=slice(start_time, end_time))
                if len(ds['time'].values) == 0:
                    print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                    continue

            fname, subsite, refdes, method, stream, deployment = cf.nc_attributes(d)

            if deployment_num is not None:
                if int(deployment.split('0')[-1]) is not deployment_num:
                    print(type(int(deployment.split('0')[-1])), type(deployment_num))
                    continue

            # get end times of deployments
            ps_df, n_streams = cf.get_preferred_stream_info(r)
            dr_data = cf.refdes_datareview_json(r)

            for index, row in ps_df.iterrows():
                deploy = row['deployment']
                deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
                if int(deploy[-4:]) not in deployments:
                    deployments.append(int(deploy[-4:]))
                if pd.to_datetime(deploy_info['stop_date']) not in end_times:
                    end_times.append(pd.to_datetime(deploy_info['stop_date']))

            data = {'lat': ds['lat'].values, 'lon': ds['lon'].values}
            new_r = pd.DataFrame(data, columns=['lat', 'lon'], index=ds['time'].values)
            sh = sh.append(new_r)


        xD = sh.lon.values
        yD = sh.lat.values
        tD = sh.index.values

        clabel = 'Time'
        ylabel = 'Latitude'
        xlabel = 'Longitude'

        fig, ax = pf.plot_profiles(xD, yD, tD, ylabel, xlabel, clabel, end_times, deployments, stdev=None)
        ax.invert_yaxis()
        ax.set_title('Glider Track - ' + r + '\n'+ 'x: platform location', fontsize=9)
        ax.set_xlim(-71.75, -69.75)
        ax.set_ylim(38.75, 40.75)
        #cbar.ax.set_yticklabels(end_times)


        # add Pioneer glider sampling area
        ax.add_patch(Rectangle((-71.5, 39.0), 1.58, 1.67, linewidth=3, edgecolor='b', facecolor='none'))
        ax.text(-71, 40.6, 'Pioneer Glider Sampling Area',
                color='blue', fontsize=8)
        # add Pioneer AUV sampling area
        # ax.add_patch(Rectangle((-71.17, 39.67), 0.92, 1.0, linewidth=3, edgecolor='m', facecolor='none'))

        array_loc = cf.return_array_subsites_standard_loc(array)

        ax.scatter(array_loc.lon, array_loc.lat, s=40, marker='x', color='k', alpha=0.3)
        #ax.legend(legn, array_loc.index, scatterpoints=1, loc='lower left', ncol=4, fontsize=8)

        pf.save_fig(save_dir, sname)

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    '''
    time option: 
    set to None if plotting all data
    set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
    '''
    start_time = None
    end_time = None
    plot_type = 'glider_track'
    deployment_num = None
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190306T174451-CP05MOAS-GL379-04-DOSTAM000-recovered_host-dosta_abcdjm_glider_recovered/catalog.html']
    main(url_list, sDir, plot_type, start_time, end_time, deployment_num)