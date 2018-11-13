#!/usr/bin/env python
"""
Created on Sep 10 2018

@author: Lori Garzio
@brief: This script identifies all data available for download listed in the OOI GUI data catalog
('https://ooinet.oceanobservatories.org/api/uframe/stream') and builds data request urls (for netCDF files) for the
science streams of the instruments input by the user. It does not check against the Datateam Database.

@usage:
sDir: location to save output
array: optional list of arrays, or an empty list if requesting all (e.g. [] or ['CP','CE'])
subsite: optional list of subsites, or an empty list if requesting all (e.g. [] or ['GI01SUMO','GA01SUMO','GS01SUMO'])
node: optional list of nodes, or an empty list if requesting all (e.g. [] or ['SBD11','SBD12'])
inst: optional list of instruments (can be partial), or an empty list if requesting all (e.g. [] or ['FLOR','CTD'])
delivery_methods: optional list of methods, or an empty list if requesting all (e.g. []  or ['streamed','telemetered','recovered'])
begin: optional start date for data request (e.g. '' or 2014-05-15T00:00:00)
end: optional end date for data request  (e.g. '' or 2015-01-01T00:00:00)
"""


import datetime as dt
import os
import pandas as pd
import requests
from . import data_request_tools
import functions.common as cf


def data_request_urls(df, begin, end):
    '''
    :return urls for data requests of science streams
    '''
    base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
    ap = '&include_annotations=true&include_provenance=true'
    url_list = []
    for i, j in df.iterrows():
        refdes = j['reference_designator']
        inst_req = data_request_tools.refdes_format(refdes)
        method = j['method']
        stream = j['stream_name']

        sys_beginTime = j['beginTime']
        sys_endTime = j['endTime']

        # check times specified against system times
        if not begin:
            beginTime = sys_beginTime
        else:
            if sys_beginTime < begin < sys_endTime:
                beginTime = begin
            else:
                print('{:s}-{:s}-{:s}: begin time entered ({:s}) is not within time ranges available in the system: ' \
                      '{:s} to {:s}'.format(refdes, method, stream, begin, sys_beginTime, sys_endTime))
                print('using system beginTime')
                beginTime = sys_beginTime

        if not end:
            endTime = sys_endTime
        else:
            if end > beginTime:
                endTime = end
            else:
                print('{:s}-{:s}-{:s}: end time entered ({:s}) is before beginTime ' \
                      '({:s})'.format(refdes, method, stream, end, sys_beginTime))
                print('using system endTime')
                endTime = sys_endTime

        url = '{:s}/{:s}{:s}/{:s}?beginDT={:s}&endDT={:s}{:s}'.format(base_url, inst_req, method, stream, beginTime, endTime, ap)
        url_list.append(url)
    return url_list


def gui_streams_science():
    r = requests.get('https://ooinet.oceanobservatories.org/api/uframe/stream')
    response = r.json()['streams']

    gui_dict_sci = {}
    for i in range(len(response)):
        try:
            method = response[i]['stream_method'].replace('-', '_')
        except AttributeError:  # skip if there is no method defined
            method = 'na'

        if response[i]['stream_dataset'] == 'Science':
            if 'bad' not in str(response[i]['stream_name']):
                gui_dict_sci[i] = {}
                refdes = response[i]['reference_designator']
                gui_dict_sci[i]['array_code'] = refdes[0:2]
                gui_dict_sci[i]['reference_designator'] = refdes
                gui_dict_sci[i]['subsite'] = refdes.split('-')[0]
                gui_dict_sci[i]['node'] = refdes.split('-')[1]
                gui_dict_sci[i]['sensor'] = refdes.split('-')[2] + '-' + refdes.split('-')[3]
                gui_dict_sci[i]['method'] = method
                gui_dict_sci[i]['stream_name'] = response[i]['stream']
                gui_dict_sci[i]['beginTime'] = response[i]['start']
                gui_dict_sci[i]['endTime'] = response[i]['end']

    gui_df_sci = pd.DataFrame.from_dict(gui_dict_sci, orient='index')

    return gui_df_sci


def main(sDir, array, subsite, node, sensor, delivery_methods, begin, end, now):
    cf.create_dir(sDir)
    begin = data_request_tools.format_date(begin)
    end = data_request_tools.format_date(end)

    # basic checks that dates were specified properly
    if end:
        if not begin:
            raise Exception('If an end date is specified, a begin date must also be specified.')

    if end:
        if begin >= end:
            raise Exception('End date entered ({:s}) is not after begin date ({:s})'.format(end, begin))

    dmethods = data_request_tools.define_methods(delivery_methods)
    gui_df_sci = gui_streams_science()
    gui_df = data_request_tools.filter_dataframe(gui_df_sci, array, subsite, node, sensor, dmethods)
    url_list = data_request_urls(gui_df, begin, end)
    urls = pd.DataFrame(url_list)
    urls.to_csv(os.path.join(sDir, 'data_request_urls_{}.csv'.format(now)), index=False, header=False)
    return url_list


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI'
    array = []  # ['CP','CE']
    subsite = []  # ['GI03FLMA','GI03FLMB']
    node = []
    inst = []  # ['CTDMO', 'FLOR']
    delivery_methods = []  # ['streamed', 'telemetered, 'recovered']
    begin = ''  # 2014-01-01T00:00:00
    end = ''  # 2015-01-01T00:00:00
    now = dt.datetime.now().strftime('%Y%m%dT%H%M')
    main(sDir, array, subsite, node, inst, delivery_methods, begin, end, now)
