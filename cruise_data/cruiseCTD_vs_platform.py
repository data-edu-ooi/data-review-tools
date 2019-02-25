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


sDir = '/Users/lgarzio/Documents/OOI/DataReviews/GA/GA01SUMO/GA01SUMO-RII11-02-DOSTAD031/cruise_compare'
CTDfile = '/Users/lgarzio/Documents/OOI/DataReviews/GA/GA01SUMO/shipboard_ctd_files/GA01SUMO_D2_nbp1510_010.cnv'
url = 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T235823-GA01SUMO-RII11-02-DOSTAD031-recovered_host-dosta_abcdjm_ctdbp_p_dcl_instrument_recovered/catalog.html'
deployment = 'D2'
uframe_window = dt.timedelta(days=1)  # dt.timedelta(hours=12)

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

ctd_pressure = np.squeeze(np.array(df['prDM']))

CTDloc = [ctd_lat, ctd_lon]  # CTD cast location

splitter = url.split('/')[-2].split('-')
refdes = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
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
            # compare the cruise CTD location against the platform location
            ploc = [np.unique(ds['lat'].values)[0], np.unique(ds['lon'].values)[0]]
            diff_loc = round(geodesic(ploc, CTDloc).kilometers, 4)
            print('The CTD cast was done {} km from the mooring location'.format(diff_loc))

            # define pressure from the data file
            if 'SBD' in ds.node:
                press = np.empty(np.shape(ds['time']))
                press[:] = 1
            else:
                press = pf.pressure_var(ds, list(ds.coords.keys()))
                if press is None:
                    press = pf.pressure_var(ds, list(ds.data_vars.keys()))
                press = ds[press].values

            if 'DOSTA' in url:
                fig, ax = plt.subplots()
                # convert oxygen from the CTD cast from ml/l to umol/kg
                # get rid of zeros and negative numbers
                ctd_do1 = np.squeeze(np.array(df['sbeox0ML/L'])) * 44.661 / 1.025
                ctd_do1[ctd_do1 <= 0.0] = np.nan
                ax.plot(ctd_do1, ctd_pressure, 'b.', markersize=3.5, label='sbeox0 (Ship CTD)')
                try:
                    ctd_do2 = np.squeeze(np.array(df['sbeox1ML/L'])) * 44.661 / 1.025
                    ctd_do2[ctd_do2 <= 0.0] = np.nan
                    ax.plot(ctd_do2, ctd_pressure, 'g.', markersize=3.5, label='sbeox1 (Ship CTD)')
                except KeyError:
                    print('No sbeox1ML/L variable in shipboard CTD file')

                pdo1 = ds['estimated_oxygen_concentration']
                pdo2 = ds['dosta_abcdjm_cspp_tc_oxygen']
                #pdo3 = ds['dissolved_oxygen']
                pdo3 = ds['dosta_analog_tc_oxygen']

                # reject nans and fill values
                ind = (~np.isnan(pdo3.values)) & (pdo3.values != pdo3._FillValue)
                if len(pdo3.values[ind]) > 0:
                    ax.plot(pdo3.values[ind], press[ind], 'r.', markersize=4, label='tc_oxygen (Platform)')
                    #ax.plot(pdo2.values[ind], press[ind], 'k.', markersize=4, label='dissolved_oxygen (Platform)')
                    ax.set_ylabel('Pressure (dbar)')
                    ax.set_xlabel('{} ({})'.format(pdo3.long_name, pdo3.units))
                    #plt.ylim([0, 50])
                    ax.invert_yaxis()
                    ax.legend(loc='best', fontsize=7)
                    ax.grid()
                    title1 = '{} vs. Shipboard CTD (Distance: {} km)'.format(refdes, diff_loc)
                    title2 = 'Cruise CTD file: {} Date: {}'.format(CTDfile.split('/')[-1],
                                                                   dt.datetime.strftime(cast_start, '%Y-%m-%dT%H:%M:%S'))
                    title3 = 'Platform: from {} to {}'.format(dt.datetime.strftime(dstart, '%Y-%m-%dT%H:%M:%S'),
                                                              dt.datetime.strftime(dend, '%Y-%m-%dT%H:%M:%S'))
                    fig.suptitle((title1 + '\n' + title2 + '\n' + title3), fontsize=8.5)
                    #sfile = '{}_{}_shipCTDcompare'.format(refdes, deployment)
                    sfile= 'test'
                    pf.save_fig(sDir, sfile)
                else:
                    print('No platform data available for Shipboard CTD time frame')

        else:
            print('No platform data available for Shipboard CTD time frame')
else:
    print('No platform data available for Shipboard CTD time frame')
