#!/usr/bin/env python
"""
Created on Mar 12 2019 by Leila Belabbassi
Modified on Apr 11 2019 by Lori Garzio
@brief plot glider tracks
"""
import os
import itertools

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import functions.plotting as pf
import functions.common as cf


def define_extent(data1, data2, version):
    dmin = np.min([np.nanmin(data1), np.nanmin(data2)])
    dmax = np.max([np.nanmax(data1), np.nanmax(data2)])

    if version == 'lon':
        diff = abs(dmax - dmin)
        if diff > 3:
            mult = 0.001
        else:
            mult = 0.0075
    else:
        mult = 0.005

    if dmin < 0:
        dmin_ex = dmin + dmin * mult
        dmax_ex = dmax - dmax * mult
    else:
        dmin_ex = dmin - dmin * mult
        dmax_ex = dmax + dmax * mult
    return dmin_ex, dmax_ex


def plot_map(save_directory, savefile, plt_title, londata, latdata, tm, array, add_box=None):
    #ax = plt.axes(projection=ccrs.PlateCarree())
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    plt.subplots_adjust(right=0.85)
    states = cfeature.NaturalEarthFeature(category="cultural", scale="10m",
                                 facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black", facecolor='grey')
    ax.add_feature(cfeature.RIVERS, zorder=10, facecolor='white')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    # gl.xlabel_style = {'size': 14.5}
    # gl.ylabel_style = {'size': 14.5}
    ax.coastlines('10m', linewidth=1)

    array_loc = cf.return_array_subsites_standard_loc(array)
    lonmin, lonmax = define_extent(array_loc.lon, londata, 'lon')
    latmin, latmax = define_extent(array_loc.lat, latdata, 'lat')

    array_loc = cf.return_array_subsites_standard_loc(array)
    sct = plt.scatter(londata, latdata, c=tm, marker='.', s=2, cmap='rainbow', transform=ccrs.Geodetic())

    ax.set_title(plt_title, fontsize=10)

    plt.scatter(array_loc.lon, array_loc.lat, s=45, marker='x', color='k')

    if add_box == 'yes':
        # add glider sampling limits
        bulk_load = pd.read_csv(
            'https://raw.githubusercontent.com/ooi-integration/asset-management/master/bulk/array_bulk_load-AssetRecord.csv')
        bulk_load['array'] = bulk_load['MIO_Inventory_Description'].str.split(' ', n=1, expand=True)[0]
        ind = bulk_load.loc[bulk_load['array'] == array].index[0]
        poly = bulk_load.iloc[ind].Array_geometry
        poly = poly.split('((')[-1].split('))')[0]
        xx = []
        yy = []
        for x in poly.split(', '):
            xx.append(float(x.split(' ')[0]))
            yy.append(float(x.split(' ')[1]))
        ax.plot(xx, yy, color='b', linewidth=2)
    else:

        ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = plt.colorbar(sct, cax=cax, label='Time')
    cbar.ax.set_yticklabels(pd.to_datetime(cbar.ax.get_yticks()).strftime(date_format='%Y-%m-%d'))

    pf.save_fig(save_directory, savefile)


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
            sname = '_'.join((r, 'glider_track'))

            sh = pd.DataFrame()
            deployments = []
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
                    #fig, ax = pf.plot_profiles(ds_lon, ds_lat, ds['time'].values, ylabel, xlabel, clabel, stdev=None)
                    plot_map(save_dir, sfile, ttl, ds_lon, ds_lat, ds['time'].values, array)

            #sh = sh.resample('H').median()  # resample hourly
            xD = sh.lon.values
            yD = sh.lat.values
            tD = sh.index.values
            title = 'Glider Track - ' + r + '\nDeployments: ' + str(deployments) + '   x: platform locations' + '\n blue box: Glider Sampling Area'
            save_dir_main = os.path.join(sDir, array, subsite, r)

            plot_map(save_dir_main, sname, title, xD, yD, tD, array, add_box='yes')


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
