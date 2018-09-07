#!/usr/bin/env python
"""
Created on Jan 11 2018
Modified on Sep 6 2018

@author: Lori Garzio
@brief: This script imports tools that compare the QC Database to the OOI GUI data catalog, builds netCDF data request
urls, and sends those requests (if prompted).

@usage:
sDir: directory where outputs are saved
username: OOI API username
token: OOI API password
"""

import datetime as dt
from tools import data_request_urls, send_data_requests_nc, data_request_tools, interactive_inputs

sDir = '/Users/lgarzio/Documents/OOI'
username = 'username'
token = 'token'

data_request_tools.create_dir(sDir)
now = dt.datetime.now().strftime('%Y%m%dT%H%M')

array, subsite, node, inst, delivery_methods = interactive_inputs.return_interactive_inputs()

begin = raw_input('Please enter a start date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''
end = raw_input('Please enter an end date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''

url_list = data_request_urls.main(sDir, array, subsite, node, inst, delivery_methods, begin, end, now)
send_data_requests_nc.main(sDir, url_list, username, token, now)
