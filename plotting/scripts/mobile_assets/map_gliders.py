#!/usr/bin/env python
"""
Created on Mar 12 2019 by Leila Belabbassi
Modified on Apr 11 2019 by Lori Garzio
@brief plot glider tracks
"""
import os
import itertools
import matplotlib
import functions.plotting as pf
import functions.common as cf
import numpy as np
import xarray as xr
import pandas as pd
matplotlib.use("TkAgg")


def main(url_list, sDir, plot_type, start_time, end_time, deployment_num):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        if 'ENG' not in r:
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

            main_sensor = r.split('-')[-1]
            fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)
            subsite = r.split('-')[0]
            array = subsite[0:2]
            save_dir = os.path.join(sDir, array, subsite, r, plot_type)
            cf.create_dir(save_dir)
            sname = '-'.join((r, 'track'))

            sh = pd.DataFrame()
            deployments = []

            clabel = 'Time'
            ylabel = 'Latitude'
            xlabel = 'Longitude'
            for ii, d in enumerate(fdatasets_sel):
                print('\nDataset {} of {}: {}'.format(ii + 1, len(fdatasets_sel), d.split('/')[-1]))
                deploy = d.split('/')[-1].split('_')[0]
                if deployment_num:
                    if int(deploy[-4:]) is not deployment_num:
                        continue

                ds = xr.open_dataset(d, mask_and_scale=False)
                ds = ds.swap_dims({'obs': 'time'})

                if start_time is not None and end_time is not None:
                    ds = ds.sel(time=slice(start_time, end_time))
                    if len(ds['time'].values) == 0:
                        print('No data to plot for specified time range: ({} to {})'.format(start_time, end_time))
                        continue

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

                if ds_lat is not None and ds_lon is not None:
                    data = {'lat': ds_lat, 'lon': ds_lon}
                    new_r = pd.DataFrame(data, columns=['lat', 'lon'], index=ds['time'].values)
                    sh = sh.append(new_r)

                    # append the deployments that are actually plotted
                    if int(deploy[-4:]) not in deployments:
                        deployments.append(int(deploy[-4:]))

                    # plot data by deployment
                    sfile = '-'.join((deploy, sname))
                    ttl = 'Glider Track - ' + r + '-' + deploy + '\nx: platform locations'
                    fig, ax = pf.plot_profiles(ds_lon, ds_lat, ds['time'].values, ylabel, xlabel, clabel, stdev=None)
                    ax.invert_yaxis()
                    ax.set_title(ttl, fontsize=9)

                    array_loc = cf.return_array_subsites_standard_loc(array)
                    ax.scatter(array_loc.lon, array_loc.lat, s=45, marker='x', color='k')
                    pf.save_fig(save_dir, sfile)

            #sh = sh.resample('H').median()  # resample hourly
            xD = sh.lon.values
            yD = sh.lat.values
            tD = sh.index.values
            title = 'Glider Track - ' + r + '\nDeployments: ' + str(deployments) + '   x: platform locations' + '\n blue box: Glider Sampling Area'

            fig, ax = pf.plot_profiles(xD, yD, tD, ylabel, xlabel, clabel, stdev=None)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=9)

            # add glider sampling limits
            bulk_load = pd.read_csv('https://raw.githubusercontent.com/ooi-integration/asset-management/master/bulk/array_bulk_load-AssetRecord.csv')
            bulk_load['array'] = bulk_load['MIO_Inventory_Description'].str.split(' ', n=1, expand=True)[0]
            ind = bulk_load.loc[bulk_load['array'] == array].index[0]
            poly = bulk_load.iloc[ind].Array_geometry
            poly = poly.split('((')[-1].split('))')[0]
            xx = []
            yy = []
            for x in poly.split(', '):
                xx.append(float(x.split(' ')[0]))
                yy.append(float(x.split(' ')[1]))

            #ax.set_xlim(-71.75, -69.75)
            #ax.set_ylim(38.75, 40.75)
            ax.plot(xx, yy, color='b', linewidth=2)
            #ax.text(np.mean(xx) - 0.15, np.max(yy) + (np.max(yy) * 0.0025), 'Glider Sampling Area', color='blue', fontsize=8)

            array_loc = cf.return_array_subsites_standard_loc(array)

            ax.scatter(array_loc.lon, array_loc.lat, s=45, marker='x', color='k')

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
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190410T154220-CE05MOAS-GL383-05-CTDGVM000-recovered_host-ctdgv_m_glider_instrument_recovered/catalog.html']
    main(url_list, sDir, plot_type, start_time, end_time, deployment_num)
