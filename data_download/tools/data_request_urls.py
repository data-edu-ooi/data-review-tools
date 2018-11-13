#!/usr/bin/env python
"""
Created on Jan 9 2018
Modified on Sep 5 2018

@author: Lori Garzio
@brief: This script compares the reference designators, methods and streams in the OOI Datateam Database
(http://ooi.visualocean.net/) to those available in the OOI GUI data catalog
('https://ooinet.oceanobservatories.org/api/uframe/stream') and builds data request urls (for netCDF files) for the
science streams of the instruments input by the user.

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
    :return urls for data requests of science streams that are found in the QC database
    '''
    base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
    ap = '&include_annotations=true&include_provenance=true'
    url_list = []
    for i, j in df.iterrows():
        if j['source'] == 'qcdb_and_gui_catalog' and j['stream_type'] == 'Science':

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


def define_source(df):
    df['source'] = ''
    df.loc[((df['in_qcdb'] == 'yes') & (df['in_gui_catalog'] == 'yes')), 'source'] = 'qcdb_and_gui_catalog'
    df.loc[((df['in_qcdb'] == 'yes') & (df['in_gui_catalog'].isnull())), 'source'] = 'qcdb_only'
    df.loc[((df['in_qcdb'].isnull()) & (df['in_gui_catalog'] == 'yes')), 'source'] = 'gui_catalog_only'
    return df


def gui_stream_list():
    r = requests.get('https://ooinet.oceanobservatories.org/api/uframe/stream')
    response = r.json()['streams']

    gui_dict_all = {}
    for i in range(len(response)):
        try:
            method = response[i]['stream_method'].replace('-', '_')
        except AttributeError:  # skip if there is no method defined
            method = 'na'

        if not response[i]['stream']:
            stream = 'no_stream'
        else:
            stream = response[i]['stream']

        gui_dict_all[i] = {}
        gui_dict_all[i]['array_name'] = response[i]['array_name']
        refdes = response[i]['reference_designator']
        gui_dict_all[i]['array_code'] = refdes[0:2]
        gui_dict_all[i]['reference_designator'] = refdes
        gui_dict_all[i]['subsite'] = refdes.split('-')[0]
        gui_dict_all[i]['node'] = refdes.split('-')[1]
        gui_dict_all[i]['sensor'] = refdes.split('-')[2] + '-' + refdes.split('-')[3]
        gui_dict_all[i]['method'] = method
        gui_dict_all[i]['stream_name'] = stream
        gui_dict_all[i]['beginTime'] = response[i]['start']
        gui_dict_all[i]['endTime'] = response[i]['end']

    gui_df_all = pd.DataFrame.from_dict(gui_dict_all, orient='index')
    gui_df_all['in_gui_catalog'] = 'yes'

    return gui_df_all


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

    db = data_request_tools.get_database()
    dbf = data_request_tools.filter_dataframe(db, array, subsite, node, sensor, dmethods)

    if dbf.empty:
        raise Exception('\nThe selected instruments/delivery_methods are not found in the QC Database.')
    else:
        gui_df_all = gui_stream_list()
        gui_df = data_request_tools.filter_dataframe(gui_df_all, array, subsite, node, sensor, dmethods)

        if gui_df.empty:
            dbf['in_gui_catalog'] = ''
            dbf['source'] = 'qcdb_only'
            dbf.to_csv(os.path.join(sDir, 'compare_qcdb_gui_catalog_{}.csv'.format(now)), index=False)
            print('\nQC Database to GUI data catalog comparison complete: %s' %os.path.join(sDir, 'compare_qcdb_gui_catalog_{}.csv'.format(now)))
            raise Exception('\nThe selected instruments are not listed in the GUI data catalog. No data requests to send.')
        else:
            merge_on = ['array_name', 'array_code', 'subsite', 'node', 'sensor', 'reference_designator', 'method', 'stream_name']
            compare = pd.merge(dbf, gui_df, on=merge_on, how='outer').sort_values(by=['reference_designator', 'method', 'stream_name'])
            compare_df = define_source(compare)
            compare_df.to_csv(os.path.join(sDir, 'compare_qcdb_gui_catalog_{}.csv'.format(now)), index=False)
            print('\nQC Database to GUI data catalog comparison complete: %s' %os.path.join(sDir, 'compare_qcdb_gui_catalog_{}.csv'.format(now)))

            url_list = data_request_urls(compare_df, begin, end)
            urls = pd.DataFrame(url_list)
            urls.to_csv(os.path.join(sDir, 'data_request_urls_{}.csv'.format(now)), index=False, header=False)
            print('\nData request urls complete: %s' % os.path.join(sDir, 'data_request_urls_{}.csv'.format(now)))

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
