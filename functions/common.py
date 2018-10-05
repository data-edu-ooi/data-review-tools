#! /usr/bin/env python
import os
import pandas as pd
import requests
import re
import itertools
import time
import xarray as xr


def check_request_status(thredds_url):
    check_complete = thredds_url.replace('/catalog/', '/fileServer/')
    check_complete = check_complete.replace('/catalog.html', '/status.txt')
    session = requests.session()
    r = session.get(check_complete)
    while r.status_code != requests.codes.ok:
        print 'Data request is still fulfilling. Trying again in 1 minute.'
        time.sleep(60)
        r = session.get(check_complete)


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


def return_raw_vars(ds_variables):
    # return a list of raw variables (eliminating engineering, qc, and timestamps)
    misc_vars = ['quality', 'string', 'timestamp', 'deployment', 'provenance', 'qc', 'time', 'mission', 'obs', 'id',
                 'serial_number', 'volt', 'ref', 'sig', 'amp', 'rph', 'calphase', 'phase', 'therm', 'description']
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
