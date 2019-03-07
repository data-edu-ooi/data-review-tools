#!/usr/bin/env python
"""
Created on Feb 21 2019

@author: Lori Garzio
@brief: Compare OOI shipboard CTD data to data downloaded from uFrame
@usage:
sDir: Directory where output files are saved
CTDfile: shipboard CTD *.cnv file
url: link to OOI THREDDs server where data downloaded from uFrame are stored
deployment: deployment or recovery to which the shipboard CTD file corresponds (e.g., D1 or R1)
uframe_window: timedelta to subset platform data (for comparison with shipboard CTD start time) (e.g. hours or days)
"""

import pandas as pd
from io import StringIO
import datetime as dt
from geopy.distance import geodesic
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import functions.common as cf
import functions.plotting as pf
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def calculate_lat_lon(sline, coord):
    sline = sline.rstrip('\r\n')
    ldeg = int(sline.split('= ')[-1].split(' ')[0])
    lmin = float(sline.split('= ')[-1].split(' ')[1])
    deg_min = ldeg + lmin/60.
    if coord == 'lat':
        if 'S' in sline.split('= ')[-1].split(' ')[-1]:
            deg_min = -deg_min
    if coord == 'lon':
        if 'W' in sline.split('= ')[-1].split(' ')[-1]:
            deg_min = -deg_min

    return deg_min


sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
CTD_dir = '/Users/lgarzio/Documents/OOI/DataReviews/CP/shipboard_ctd_files'
platform = 'CP04OSSM'
urls = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T162623-CP04OSSM-MFD37-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html',
        'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181218T162658-CP04OSSM-RID27-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html']
uframe_windows = [dt.timedelta(days=1), dt.timedelta(hours=6)]  # dt.timedelta(hours=12)
#uframe_windows = [dt.timedelta(days=1)]

CTD_files = []
for file in os.listdir(CTD_dir):
    if file.startswith(platform):
        CTD_files.append(os.path.join(CTD_dir, file))

for uframe_window in uframe_windows:
    print(uframe_window)
    for CTDfile in CTD_files:
        print(CTDfile)
        deployment = CTDfile.split('/')[-1].split('_')[1]
        CTDfile_platform = CTDfile.split('/')[-1].split('_')[0]
        ff = open(CTDfile, 'rb')
        ff = StringIO(ff.read().decode(encoding='utf-8', errors='replace'))
        header1, header2 = [], []
        for k, line in enumerate(ff.readlines()):
            if '* NMEA Latitude' in line:
                ctd_lat = calculate_lat_lon(line, 'lat')
            if '* NMEA Longitude' in line:
                ctd_lon = calculate_lat_lon(line, 'lon')
            if '# start_time' in line:
                line = line.rstrip('\r\n')
                cast_start = dt.datetime.strptime(line.split('= ')[1].split(' [')[0], '%b %d %Y %H:%M:%S')

            if '# name ' in line:
                line = line.rstrip('\r\n')
                name, desc = line.split('=')[1].split(':')
                if name in [u' sigma-\ufffd00', u' sigma-\ufffd11']:
                    name = 'sigma'
                header1.append(str(name).lstrip())
                header2.append(str(desc).lstrip())

            if '*END*' in line:
                skiprows = k + 1

        ff.seek(0)
        df = pd.read_table(ff, header=None, names=header1, index_col=None, skiprows=skiprows, delim_whitespace=True)
        df.columns = pd.MultiIndex.from_tuples(zip(df.columns, header2))
        ff.close()

        try:
            ctd_pressure = np.squeeze(np.array(df['prDM']))
        except KeyError:
            try:
                ctd_pressure = np.squeeze(np.array(df['PRES']))
            except KeyError:
                print('No pressure variable found in the cruise CTD file')
                ctd_pressure = []

        CTDloc = [ctd_lat, ctd_lon]  # CTD cast location

        for url in urls:
            splitter = url.split('/')[-2].split('-')
            refdes = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if CTDfile_platform in refdes:  # if the file is mapped to this reference designator
                print(refdes)
                catalog_rms = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4], splitter[5], splitter[6]))
                ud = cf.get_nc_urls([url])
                datasets = []
                for u in ud:
                    spl = u.split('/')[-1].split('_')
                    file_rms = '_'.join(spl[1:-1])
                    if catalog_rms == file_rms and deployment[-1] in spl[0]:
                        datasets.append(u)

                if len(datasets) == 1:
                    with xr.open_dataset(datasets[0], mask_and_scale=False) as ds:
                        ds = ds.swap_dims({'obs': 'time'})
                        # select data within +/- 1 day of the CTD cast
                        dstart = cast_start - uframe_window
                        dend = cast_start + uframe_window
                        ds = ds.sel(time=slice(dstart, dend))
                        if len(ds['time']) > 0:
                            save_dir = os.path.join(sDir, refdes[0:2], refdes[0:8], refdes, 'cruise_compare')
                            cf.create_dir(save_dir)
                            # compare the cruise CTD location against the platform location
                            if 'MOAS' in ds.subsite:
                                lat = ds['lat'].values
                                lat_ind = ~np.isnan(lat)
                                lon = ds['lon'].values
                                lon_ind = ~np.isnan(lon)
                                ploc = [np.median(lat[lat_ind]), np.median(lon[lon_ind])]
                            else:
                                ploc = [np.unique(ds['lat'].values)[0], np.unique(ds['lon'].values)[0]]
                            diff_loc = round(geodesic(ploc, CTDloc).kilometers, 4)
                            print('The CTD cast was done {} km from the mooring location'.format(diff_loc))

                            # define pressure from the data file
                            if 'SBD' in ds.node:
                                press = np.empty(np.shape(ds['time']))
                                press[:] = 1
                            elif 'RID' in ds.node:
                                press = np.empty(np.shape(ds['time']))
                                press[:] = 7
                            # elif 'RIS' in ds.node:
                            #     press = np.empty(np.shape(ds['time']))
                            #     press[:] = 30
                            else:
                                press = pf.pressure_var(ds, list(ds.coords.keys()))
                                if press is None:
                                    press = pf.pressure_var(ds, list(ds.data_vars.keys()))
                                press = ds[press].values

                            if 'CTD' in ds.sensor:
                                try:
                                    ctd_cond = np.squeeze(np.array(df['CNDC']))
                                except KeyError:
                                    try:
                                        ctd_cond = np.squeeze(np.array(df['c1mS/cm'])) / 10
                                    except KeyError:
                                        try:
                                            ctd_cond = np.squeeze(np.array(df['c0S/m']))
                                        except KeyError:
                                            print('No conductivity variable found in the cruise CTD file')
                                            ctd_cond = []

                                try:
                                    ctd_den = np.squeeze(np.array(df['density']))
                                except KeyError:
                                    try:
                                        ctd_den = np.squeeze(np.array(df['sigma-\xe900'])) + 1000
                                    except KeyError:
                                        try:
                                            ctd_den = np.squeeze(np.array(df['density00']))
                                        except KeyError:
                                            print('No density variable found in the cruise CTD file')
                                            ctd_den = []

                                try:
                                    ctd_temp = np.squeeze(np.array(df['TEMP']))
                                except KeyError:
                                    try:
                                        ctd_temp = np.squeeze(np.array(df['t090C']))
                                    except KeyError:
                                        print('No temperature variable found in the cruise CTD file')
                                        ctd_temp = []

                                try:
                                    ctd_sal = np.squeeze(np.array(df['PSAL']))
                                except KeyError:
                                    try:
                                        ctd_sal = np.squeeze(np.array(df['sal00']))
                                    except KeyError:
                                        print('No salinity variable found in the cruise CTD file')
                                        ctd_sal = []

                                ctd_vars = [ctd_cond, ctd_den, ctd_temp, ctd_sal]
                                if 'MOAS' in ds.subsite:
                                    platform_vars = ['sci_water_cond', 'sci_seawater_density', 'sci_water_temp',
                                                     'practical_salinity']
                                else:
                                    platform_vars = ['ctdpf_ckl_seawater_conductivity', 'density',
                                                     'ctdpf_ckl_seawater_temperature', 'practical_salinity']

                                for i in range(len(ctd_vars)):
                                    fig, ax = plt.subplots()
                                    ctdvar = ctd_vars[i]
                                    if len(ctdvar) > 0:
                                        ctdvar[ctdvar <= 0.0] = np.nan
                                        ax.plot(ctdvar, ctd_pressure, 'b.', markersize=3.5, label='Ship CTD')

                                        pvarname = platform_vars[i]
                                        pvar = ds[pvarname]

                                        pvar[pvar <= 0.0] = np.nan  # get rid of zeros and negative numbers
                                        if 'density' in pvarname:
                                            pvar[pvar <= 1000.0] = np.nan  # get rid of zeros and negative numbers
                                        # reject nans and fill values
                                        ind = (~np.isnan(pvar.values)) & (pvar.values != pvar._FillValue)
                                        if len(pvar.values[ind]) > 0:
                                            ax.plot(pvar.values[ind], press[ind], 'r.', markersize=4,
                                                    label='Platform')
                                            ax.set_ylabel('Pressure (dbar)')
                                            ax.set_xlabel('{} ({})'.format(pvar.long_name, pvar.units))
                                            # plt.ylim([0, 50])
                                            ax.invert_yaxis()
                                            ax.legend(loc='best', fontsize=7)
                                            ax.grid()
                                            title1 = '{} vs. Shipboard CTD (Distance: {} km)'.format(refdes, diff_loc)
                                            title2 = 'Cruise CTD file: {} Date: {}'.format(CTDfile.split('/')[-1],
                                                                                           dt.datetime.strftime(cast_start,
                                                                                                                '%Y-%m-%dT%H:%M:%S'))
                                            title3 = 'Platform: from {} to {}'.format(str(ds['time'].values[0])[:19],
                                                                                      str(ds['time'].values[-1])[:19])
                                            fig.suptitle((title1 + '\n' + title2 + '\n' + title3), fontsize=8.5)
                                            sfile = '{}_{}_shipCTDcompare_{}'.format(refdes, deployment, pvarname)
                                            pf.save_fig(save_dir, sfile)
                                            plt.close()
                                        else:
                                            print('No platform data available for Shipboard CTD time frame')

                            if 'FLOR' in ds.sensor:
                                fig, ax = plt.subplots()
                                ctd_chl = np.squeeze(np.array(df['flECO-AFL']))
                                ctd_chl[ctd_chl <= 0.0] = np.nan
                                ax.plot(ctd_chl, ctd_pressure, 'b.', markersize=3.5, label='flECO-AFL (Ship CTD)')

                                if 'MOAS' in ds.subsite:
                                    if 'FLORTM' in ds.sensor:
                                        chlname = 'sci_flbbcd_chlor_units'
                                    else:
                                        chlname = 'sci_flbb_chlor_units'
                                else:
                                    chlname = 'fluorometric_chlorophyll_a'
                                pchla = ds[chlname]

                                pchla[pchla <= 0.0] = np.nan  # get rid of zeros and negative numbers

                                # reject nans and fill values
                                ind = (~np.isnan(pchla.values)) & (pchla.values != pchla._FillValue)
                                if len(pchla.values[ind]) > 0:
                                    ax.plot(pchla.values[ind], press[ind], 'r.', markersize=4,
                                            label='{} (Platform)'.format(chlname))
                                    ax.set_ylabel('Pressure (dbar)')
                                    ax.set_xlabel('{} ({})'.format(pchla.long_name, pchla.units))
                                    # plt.ylim([0, 50])
                                    ax.invert_yaxis()
                                    ax.legend(loc='best', fontsize=7)
                                    ax.grid()
                                    title1 = '{} vs. Shipboard CTD (Distance: {} km)'.format(refdes, diff_loc)
                                    title2 = 'Cruise CTD file: {} Date: {}'.format(CTDfile.split('/')[-1],
                                                                                   dt.datetime.strftime(cast_start,
                                                                                                        '%Y-%m-%dT%H:%M:%S'))
                                    title3 = 'Platform: from {} to {}'.format(str(ds['time'].values[0])[:19],
                                                                              str(ds['time'].values[-1])[:19])
                                    fig.suptitle((title1 + '\n' + title2 + '\n' + title3), fontsize=8.5)
                                    sfile = '{}_{}_shipCTDcompare_{}'.format(refdes, deployment, chlname)
                                    pf.save_fig(save_dir, sfile)
                                    plt.close()
                                else:
                                    print('No platform data available for Shipboard CTD time frame')

                            if 'MOAS' in ds.subsite and 'DOSTA' in ds.sensor:
                                # convert oxygen from the CTD cast from ml/l to umol/L (*44.661) then umol/kg (/1.025)
                                ctd_do1_l = np.squeeze(np.array(df['sbeox0ML/L'])) * 44.661
                                ctd_do1_kg = ctd_do1_l / 1.025

                                ctd_vars = [ctd_do1_l, ctd_do1_kg]

                                platform_vars = ['sci_oxy4_oxygen', 'sci_abs_oxygen']

                                for i in range(len(ctd_vars)):
                                    fig, ax = plt.subplots()
                                    ctdvar = ctd_vars[i]
                                    if len(ctdvar) > 0:
                                        ctdvar[ctdvar <= 0.0] = np.nan
                                        ax.plot(ctdvar, ctd_pressure, 'b.', markersize=3.5, label='Ship CTD')

                                        pvarname = platform_vars[i]
                                        pvar = ds[pvarname]

                                        pvar[pvar <= 0.0] = np.nan  # get rid of zeros and negative numbers
                                        if 'density' in pvarname:
                                            pvar[pvar <= 1000.0] = np.nan  # get rid of zeros and negative numbers
                                        # reject nans and fill values
                                        ind = (~np.isnan(pvar.values)) & (pvar.values != pvar._FillValue)
                                        if len(pvar.values[ind]) > 0:
                                            ax.plot(pvar.values[ind], press[ind], 'r.', markersize=4,
                                                    label='Platform')
                                            ax.set_ylabel('Pressure (dbar)')
                                            ax.set_xlabel('{} ({})'.format(pvar.long_name, pvar.units))
                                            # plt.ylim([0, 50])
                                            ax.invert_yaxis()
                                            ax.legend(loc='best', fontsize=7)
                                            ax.grid()
                                            title1 = '{} vs. Shipboard CTD (Distance: {} km)'.format(refdes, diff_loc)
                                            title2 = 'Cruise CTD file: {} Date: {}'.format(CTDfile.split('/')[-1],
                                                                                           dt.datetime.strftime(cast_start,
                                                                                                                '%Y-%m-%dT%H:%M:%S'))
                                            title3 = 'Platform: from {} to {}'.format(str(ds['time'].values[0])[:19],
                                                                                      str(ds['time'].values[-1])[:19])
                                            fig.suptitle((title1 + '\n' + title2 + '\n' + title3), fontsize=8.5)
                                            sfile = '{}_{}_shipCTDcompare_{}'.format(refdes, deployment, pvarname)
                                            pf.save_fig(save_dir, sfile)
                                            plt.close()
                                        else:
                                            print('No platform data available for Shipboard CTD time frame')

                            if 'MOAS' not in ds.subsite:
                                if 'DOSTA' in ds.sensor or 'DOFST' in ds.sensor:
                                    # convert oxygen from the CTD cast from ml/l to umol/L (*44.661)
                                    # get rid of zeros and negative numbers
                                    ctd_do1 = np.squeeze(np.array(df['sbeox0ML/L'])) * 44.661
                                    ctd_do1[ctd_do1 <= 0.0] = np.nan

                                    try:
                                        ctd_do2 = np.squeeze(np.array(df['sbeox1ML/L'])) * 44.661
                                        ctd_do2[ctd_do2 <= 0.0] = np.nan
                                    except KeyError:
                                        ctd_do2 = []
                                        print('No sbeox1ML/L variable in shipboard CTD file')

                                    try:
                                        ctd_do3 = np.squeeze(np.array(df['sbeox0ML/L.1'])) * 44.661
                                        ctd_do3[ctd_do3 <= 0.0] = np.nan
                                    except KeyError:
                                        ctd_do3 = []
                                        print('No sbeox0ML/L.1 variable in shipboard CTD file')

                                    if 'DOSTA' in ds.sensor:
                                        platform_vars = ['estimated_oxygen_concentration', 'dissolved_oxygen']
                                    elif 'DOFST' in ds.sensor:
                                        platform_vars = ['dofst_k_oxygen_l2']

                                    for var in platform_vars:
                                        fig, ax = plt.subplots()
                                        pdo = ds[var]
                                        pdo[pdo <= 0.0] = np.nan  # get rid of zeros and negative numbers

                                        # reject nans and fill values
                                        ind = (~np.isnan(pdo.values)) & (pdo.values != pdo._FillValue)
                                        if len(pdo.values[ind]) > 0:
                                            # when plotting dissolved_oxygen (umol/kg), convert shipboard CTD to umol/kg
                                            if var == 'dissolved_oxygen' or var == 'dofst_k_oxygen_l2':
                                                ctd_do1 = ctd_do1 / 1.025
                                                if len(ctd_do2) > 0:
                                                    ctd_do2 = ctd_do2 / 1.025
                                                if len(ctd_do3) > 0:
                                                    ctd_do3 = ctd_do3 / 1.025
                                            ax.plot(ctd_do1, ctd_pressure, 'b.', markersize=3.5, label='sbeox0 (Ship CTD)')
                                            if len(ctd_do2) > 0:
                                                ax.plot(ctd_do2, ctd_pressure, 'g.', markersize=3.5, label='sbeox1 (Ship CTD)')
                                            if len(ctd_do3) > 0:
                                                ax.plot(ctd_do3, ctd_pressure, 'g.', markersize=3.5, label='sbeox0.1 (Ship CTD)')
                                            ax.plot(pdo.values[ind], press[ind], 'r.', markersize=4, label='{} (Platform)'.format(var))
                                            #ax.plot(pdo2.values[ind], press[ind], 'k.', markersize=4, label='dissolved_oxygen (Platform)')
                                            ax.set_ylabel('Pressure (dbar)')
                                            ax.set_xlabel('{} ({})'.format(pdo.long_name, pdo.units))
                                            #plt.ylim([0, 50])
                                            ax.invert_yaxis()
                                            ax.legend(loc='best', fontsize=7)
                                            ax.grid()
                                            title1 = '{} vs. Shipboard CTD (Distance: {} km)'.format(refdes, diff_loc)
                                            title2 = 'Cruise CTD file: {} Date: {}'.format(CTDfile.split('/')[-1],
                                                                                           dt.datetime.strftime(cast_start, '%Y-%m-%dT%H:%M:%S'))
                                            title3 = 'Platform: from {} to {}'.format(str(ds['time'].values[0])[:19],
                                                                                      str(ds['time'].values[-1])[:19])
                                            fig.suptitle((title1 + '\n' + title2 + '\n' + title3), fontsize=8.5)
                                            sfile = '{}_{}_shipCTDcompare_{}'.format(refdes, deployment, var)
                                            pf.save_fig(save_dir, sfile)
                                            plt.close()
                                        else:
                                            print('No platform data available for Shipboard CTD time frame')

                        else:
                            print('No platform data available for Shipboard CTD time frame')
            else:
                print('No platform data available for Shipboard CTD time frame')
