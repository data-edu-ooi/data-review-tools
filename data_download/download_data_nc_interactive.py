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
import functions.common as cf
import scripts

sDir = '/Users/lgarzio/Documents/OOI'
username = 'username'
token = 'token'

cf.create_dir(sDir)
now = dt.datetime.now().strftime('%Y%m%dT%H%M')

array, subsite, node, inst, delivery_methods = scripts.interactive_inputs.return_interactive_inputs()

begin = input('Please enter a start date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''
end = input('Please enter an end date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''

url_list = scripts.data_request_urls.main(sDir, array, subsite, node, inst, delivery_methods, begin, end, now)
thredds_output_urls = scripts.send_data_requests_nc.main(sDir, url_list, username, token, now)

print('Seeing if the requests have fulfilled...')
for i in range(len(thredds_output_urls)):
    url = thredds_output_urls[i]
    print('\nDataset {} of {}: {}'.format((i + 1), len(thredds_output_urls), url))
    cf.check_request_status(url)
