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
import functions.common as cf
import scripts

sDir = '/Users/lgarzio/Documents/OOI'
username = 'username'
token = 'token'

cf.create_dir(sDir)
now = dt.datetime.now().strftime('%Y%m%dT%H%M')

arrays = input('\nPlease select arrays (CE, CP, GA, GI, GP, GS, RS). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
array = scripts.data_request_tools.format_inputs(arrays)

subsites = input('\nPlease fully-qualified subsites (e.g. GI01SUMO, GI05MOAS). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
subsite = scripts.data_request_tools.format_inputs(subsites)

nodes = input('\nPlease select fully-qualified nodes. (e.g. GL469, GL477). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
node = scripts.data_request_tools.format_inputs(nodes)

insts = input('\nPlease select instruments (can be partial (e.g. CTD) or fully-qualified (e.g. 04-CTDGVM000)). Must be comma separated (if choosing multiple) or press enter to select all: ') or ''
inst = scripts.data_request_tools.format_inputs(insts)

delivery_methods = input('\nPlease select valid delivery methods [recovered, telemetered, streamed]. Must be comma separated (if choosing multiple) or press enter to select all: ') or ''

begin = input('Please enter a start date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''
end = input('Please enter an end date for your data requests with format <2014-01-01T00:00:00> or press enter to request all available data: ') or ''

url_list = scripts.data_request_urls_nocheck.main(sDir, array, subsite, node, inst, delivery_methods, begin, end, now)
thredds_urls = scripts.send_data_requests_nc.main(sDir, url_list, username, token, now)

scripts.thredds_download_nc.main(sDir, thredds_urls)
