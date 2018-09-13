#!/usr/bin/env python
"""
Created on Sep 10 2018

@author: Lori Garzio
@brief: This script imports tools that build netCDF data request urls for science streams of the instruments input by
the user (does not check against the Datateam Database), sends those requests (if prompted), and downloads the netCDF,
provenance, and annotation files to a local machine.

@usage:
sDir: directory where outputs are saved
username: OOI API username
token: OOI API password
"""

import datetime as dt
from tools import data_request_urls_nocheck, send_data_requests_nc, thredds_download_nc, data_request_tools

sDir = '/Users/lgarzio/Documents/OOI'
username = 'username'
token = 'token'

data_request_tools.create_dir(sDir)
now = dt.datetime.now().strftime('%Y%m%dT%H%M')

arrays = raw_input('\nPlease select arrays (CE, CP, GA, GI, GP, GS, RS). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
array = data_request_tools.format_inputs(arrays)

subsites = raw_input('\nPlease fully-qualified subsites (e.g. GI01SUMO, GI05MOAS). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
subsite = data_request_tools.format_inputs(subsites)

nodes = raw_input('\nPlease select fully-qualified nodes. (e.g. GL469, GL477). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
node = data_request_tools.format_inputs(nodes)

insts = raw_input('\nPlease select instruments (can be partial (e.g. CTD) or fully-qualified (e.g. 04-CTDGVM000)). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
inst = data_request_tools.format_inputs(insts)

delivery_methods = raw_input('\nPlease select valid delivery methods [recovered, telemetered, streamed]. Must be comma separated (if choosing multiple) or press enter to select all: ') or ''

begin = raw_input('Please enter a start date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''
end = raw_input('Please enter an end date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''

url_list = data_request_urls_nocheck.main(sDir, array, subsite, node, inst, delivery_methods, begin, end, now)
thredds_urls = send_data_requests_nc.main(sDir, url_list, username, token, now)

thredds_download_nc.main(sDir, thredds_urls)
