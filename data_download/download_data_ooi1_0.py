#!/usr/bin/env python
"""
Created  on Sep 7 2018

@author: Lori Garzio
@brief: This script imports tools to use the data_review_list
(https://github.com/data-edu-ooi/data-review-tools/tree/master/review_list) to download OOI 1.0 datasets via the OOI
M2M interface.

@usage:
sDir: directory where outputs are saved, and where f is located
f: optional csv file containing data to download, columns: array, subsite, node, sensor, delivery_method,
reference_designator (entires in all columns are optional). If this file is not provided, the script will prompt
the user for inputs
username: OOI API username
token: OOI API password
"""


import datetime as dt
import pandas as pd
import os
import itertools
from tools import data_request_tools, interactive_inputs, data_request_urls_ooi1_0, send_data_requests_nc

sDir = '/Users/lgarzio/Documents/OOI/test'
f = ''  # optional i.e. 'data_download.csv'
username = 'OOIAPI-BJOX1E3EIP431N'
token = 'MS122F2C06J1V7'

data_request_tools.create_dir(sDir)
now = dt.datetime.now().strftime('%Y%m%dT%H%M')

if not f:
    array, subsite, node, inst, delivery_methods = interactive_inputs.return_interactive_inputs()
    f_url_list = data_request_urls_ooi1_0.main(sDir, array, subsite, node, inst, delivery_methods, now)
else:
    df = pd.read_csv(os.path.join(sDir, f))
    url_list = []
    for i, j in df.iterrows():
        array = data_request_tools.check_str(j['array'])
        array = data_request_tools.format_inputs(array)
        refdes = j['reference_designator']
        if type(refdes) == str:
            subsite = data_request_tools.format_inputs(refdes.split('-')[0])
            node = data_request_tools.format_inputs(refdes.split('-')[1])
            inst = data_request_tools.format_inputs('-'.join((refdes.split('-')[2], refdes.split('-')[3])))
        else:
            subsite = data_request_tools.check_str(j['subsite'])
            subsite = data_request_tools.format_inputs(subsite)
            node = data_request_tools.check_str(j['node'])
            node = data_request_tools.format_inputs(node)
            inst = data_request_tools.check_str(j['sensor'])
            inst = data_request_tools.format_inputs(inst)
        delivery_methods = data_request_tools.check_str(j['delivery_method'])
        delivery_methods = data_request_tools.format_inputs(delivery_methods)

        urls = data_request_urls_ooi1_0.main(sDir, array, subsite, node, inst, delivery_methods, now)
        url_list.append(urls)

    f_url_list = list(itertools.chain(*url_list))

send_data_requests_nc.main(sDir, f_url_list, username, token, now)
