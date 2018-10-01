#! /usr/bin/env python
import os
import pandas as pd
import requests
import numpy as np
import re
import itertools


def get_nc_urls(catalog_urls):
    """
    Return a list of urls to access netCDF files in THREDDS
    :param urls: List of THREDDS catalog urls
    :return: List of netCDF urls for data access
    """
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
    datasets = []
    for i in catalog_urls:
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


def get_global_ranges(platform, node, sensor, variable, api_user=None, api_token=None):
    port = '12578'
    base_url = '{}/qcparameters/inv/{}/{}/{}/'.format(port, platform, node, sensor)
    url = 'https://ooinet.oceanobservatories.org/api/m2m/{}'.format(base_url)
    if (api_user is None) or (api_token is None):
        r = requests.get(url, verify=False)
    else:
        r = requests.get(url, auth=(api_user, api_token), verify=False)

    if r.status_code is 200:
        if r.json(): # If r.json is not empty
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
