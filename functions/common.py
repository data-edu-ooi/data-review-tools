#! /usr/bin/env python
import os
import pandas as pd
import requests
import re
import itertools
import time
import xarray as xr
import numpy as np
from geopy.distance import geodesic


def check_request_status(thredds_url):
    check_complete = thredds_url.replace('/catalog/', '/fileServer/')
    check_complete = check_complete.replace('/catalog.html', '/status.txt')
    session = requests.session()
    r = session.get(check_complete)
    while r.status_code != requests.codes.ok:
        print 'Data request is still fulfilling. Trying again in 1 minute.'
        time.sleep(60)
        r = session.get(check_complete)
    print 'Data request has fulfilled.'


def create_dir(new_dir):
    # Check if dir exists.. if it doesn't... create it.
    if not os.path.isdir(new_dir):
        try:
            os.makedirs(new_dir)
        except OSError:
            if os.path.exists(new_dir):
                pass
            else:
                raise


def deploy_location_check(refdes):
    # Calculate the distance in kilometers between an instrument's location (defined in asset management) and previous
    # deployment locations
    deploy_loc = {}
    dr_data = refdes_datareview_json(refdes)
    for i, d in enumerate(dr_data['instrument']['deployments']):
        deploy_loc[i] = {}
        deploy_loc[i]['deployment'] = d['deployment_number']
        deploy_loc[i]['lat'] = d['latitude']
        deploy_loc[i]['lon'] = d['longitude']

    # put info in a data frame
    df = pd.DataFrame.from_dict(deploy_loc, orient='index').sort_index()
    y = {}
    for i, k in df.iterrows():
        if i > 0:
            loc1 = [k['lat'], k['lon']]
            d1 = int(k['deployment'])
            for x in range(i):
                info0 = df.iloc[x]
                compare = 'diff_km_D{}_to_D{}'.format(d1, int(info0['deployment']))
                loc0 = [info0['lat'], info0['lon']]
                diff_loc = round(geodesic(loc0, loc1).kilometers, 4)
                y.update({compare: diff_loc})
    return y


def filter_collocated_instruments(main_sensor, datasets):
    # Remove collocated instruments from a list of datasets from THREDDS
    datasets_filtered = []
    for d in datasets:
        fname = d.split('/')[-1]
        if main_sensor in fname:
            datasets_filtered.append(d)
    return datasets_filtered


def get_global_ranges(platform, node, sensor, variable, api_user=None, api_token=None):
    port = '12578'
    base_url = '{}/qcparameters/inv/{}/{}/{}/'.format(port, platform, node, sensor)
    url = 'https://ooinet.oceanobservatories.org/api/m2m/{}'.format(base_url)
    if (api_user is None) or (api_token is None):
        r = requests.get(url, verify=False)
    else:
        r = requests.get(url, auth=(api_user, api_token), verify=False)

    if r.status_code is 200:
        if r.json():  # If r.json is not empty
            values = pd.io.json.json_normalize(r.json())
            t1 = values[values['qcParameterPK.streamParameter'] == variable]
            if not t1.empty:
                t2 = t1[t1['qcParameterPK.qcId'] == 'dataqc_globalrangetest_minmax']
                if not t2.empty:
                    local_min = float(t2[t2['qcParameterPK.parameter'] == 'dat_min'].iloc[0]['value'])
                    local_max = float(t2[t2['qcParameterPK.parameter'] == 'dat_max'].iloc[0]['value'])
                else:
                    local_min = None
                    local_max = None
            else:
                local_min = None
                local_max = None
        else:
            local_min = None
            local_max = None
    else:
        local_min = None
        local_max = None
    return [local_min, local_max]


def get_nc_urls(catalog_urls):
    """
    Return a list of urls to access netCDF files in THREDDS
    :param catalog_urls: List of THREDDS catalog urls
    :return: List of netCDF urls for data access
    """
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
    datasets = []
    for i in catalog_urls:
        # check that the request has fulfilled
        check_request_status(i)

        dataset = requests.get(i).text
        ii = re.findall(r'href=[\'"]?([^\'" >]+)', dataset)
        x = re.findall(r'(ooi/.*?.nc)', dataset)
        for i in x:
            if i.endswith('.nc') == False:
                x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        dataset = [os.path.join(tds_url, i) for i in x]
        datasets.append(dataset)
    datasets = list(itertools.chain(*datasets))
    return datasets


def nc_attributes(nc_file):
    """
    Return global information from a netCDF file
    :param nc_file: url for a netCDF file on the THREDDs server
    """
    with xr.open_dataset(nc_file) as ds:
        fname = nc_file.split('/')[-1].split('.nc')[0].split('.')[0]
        subsite = ds.subsite
        node = ds.node
        sensor = ds.sensor
        refdes = '-'.join((subsite, node, sensor))
        method = ds.collection_method
        stream = ds.stream
        deployment = fname[0:14]

    return fname, subsite, refdes, method, stream, deployment


def refdes_datareview_json(refdes):
    # returns information about a reference designator from the Data Review Database
    url = 'http://datareview.marine.rutgers.edu/instruments/view/'
    ref_des_url = os.path.join(url, refdes)
    ref_des_url += '.json'
    r = requests.get(ref_des_url).json()
    return r


def reject_extreme_values(data):
    # Reject extreme values
    return (data > -1e6) & (data < 1e6)


def reject_outliers(data, m=3):
    """
    Reject outliers beyond m standard deviations of the mean.
    :param data: numpy array containing data
    :param m: number of standard deviations from the mean. Default: 3
    """
    return abs(data - np.nanmean(data)) < m * np.nanstd(data)


def return_stream_vars(stream):
    # return all variables that should be found in a stream (from the data review database)
    stream_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        stream_vars.append(p['name'])
    return stream_vars


def return_raw_vars(ds_variables):
    # return a list of raw variables (eliminating engineering, qc, and timestamps)
    misc_vars = ['quality', 'string', 'timestamp', 'deployment', 'provenance', 'qc', 'time', 'mission', 'obs', 'id',
                 'serial_number', 'volt', 'ref', 'sig', 'amp', 'rph', 'calphase', 'phase', 'therm', 'description',
                 'lat', 'lon']
    reg_ex = re.compile('|'.join(misc_vars))
    raw_vars = [s for s in ds_variables if not reg_ex.search(s)]
    return raw_vars


def return_science_vars(stream):
    # return only the science variables (defined in preload) for a data stream
    sci_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        if p['data_product_type'] == 'Science Data':
            sci_vars.append(p['name'])
    return sci_vars


def variable_statistics(variable, stdev=None):
    """
    Calculate statistics for a variable of interest
    :param variable: array containing data
    :param stdev: desired standard deviation to exclude from analysis
    """
    if stdev is None:
        varD = variable.data
        num_outliers = 0
    else:
        ind = reject_extreme_values(variable)
        var = variable[ind]

        ind2 = reject_outliers(var, stdev)
        varD = var[ind2].data
        num_outliers = len(variable) - len(varD)

    mean = round(np.nanmean(varD), 4)
    min = round(np.nanmin(varD), 4)
    max = round(np.nanmax(varD), 4)
    sd = round(np.nanstd(varD), 4)
    n = len(varD)

    return [num_outliers, mean, min, max, sd, n]