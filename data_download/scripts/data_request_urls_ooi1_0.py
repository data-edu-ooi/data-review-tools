#!/usr/bin/env python
"""
Created  on Sep 6 2018

@author: Lori Garzio
@brief: This script uses the data_review_list.csv and the OOI Datateam Database
(http://ooi.visualocean.net/) to build data request urls (for netCDF files) for data download - only for science streams
for instrument deployments during OOI 1.0.

@usage:
sDir: location to save output
array: optional list of arrays, or an empty list if requesting all (e.g. [] or ['CP','CE'])
subsite: optional list of subsites, or an empty list if requesting all (e.g. [] or ['GI01SUMO','GA01SUMO','GS01SUMO'])
node: optional list of nodes, or an empty list if requesting all (e.g. [] or ['SBD11','SBD12'])
inst: optional list of instruments (can be partial), or an empty list if requesting all (e.g. [] or ['FLOR','CTD'])
delivery_methods: optional list of methods, or an empty list if requesting all (e.g. []  or ['streamed','telemetered','recovered'])
"""


import datetime as dt
import os
import pandas as pd
import csv
from . import data_request_tools
import functions.common as cf


def build_data_request_urls(df):
    '''
    :return urls for data requests of science streams that are found in the QC database
    '''
    base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
    ap = '&include_annotations=true&include_provenance=true'
    url_list = []
    for i, j in df.iterrows():
        if j['stream_type'] == 'Science':

            refdes = j['reference_designator']
            rd = refdes.split('-')
            inst_req = '{:s}/{:s}/{:s}-{:s}/'.format(rd[0], rd[1], rd[2], rd[3])
            method = j['method']
            stream = j['stream_name']
            beginTime = j['begin']
            endTime = j['end']

            url = '{:s}/{:s}{:s}/{:s}?beginDT={:s}&endDT={:s}{:s}'.format(base_url, inst_req, method, stream, beginTime, endTime, ap)
            url_list.append(url)
    return url_list


def main(sDir, array, subsite, node, inst, delivery_methods, now=dt.datetime.now().strftime('%Y%m%dT%H%M')):
    cf.create_dir(sDir)
    rl = pd.read_csv('https://raw.githubusercontent.com/data-edu-ooi/data-review-tools/master/review_list/data_review_list.csv')

    dmethods = data_request_tools.define_methods(delivery_methods)
    rlf = data_request_tools.filter_dataframe(rl, array, subsite, node, inst)
    refdes_list = rlf['Reference Designator'].unique().tolist()

    output = dict()
    for i, r in enumerate(refdes_list):
        rlf_refdes = rlf.loc[((rlf['Reference Designator'] == r) & (rlf['status'] == 'for review'))]
        if not rlf_refdes.empty:
            output[i] = {}
            output[i]['reference_designator'] = r
            output[i]['begin'] = data_request_tools.format_date(min(rlf_refdes['startDateTime']))
            output[i]['end'] = data_request_tools.format_date(max(rlf_refdes['stopDateTime']))
            output[i]['deployments'] = map(int, rlf_refdes['deploymentNumber'].tolist())

    output_df = pd.DataFrame.from_dict(output, orient='index')

    fpath = os.path.join(sDir, 'data_review_dates_deployments.csv')
    if os.path.isfile(fpath):
        with open(fpath, 'a') as ff:
            writer = csv.writer(ff)
            for i in output:
                writer.writerow([output[i]['reference_designator'], output[i]['begin'], output[i]['end'],
                                 str(output[i]['deployments'])])
    else:
        output_df.to_csv(fpath, index=False)

    db = data_request_tools.get_database()
    dbf = data_request_tools.filter_dataframe(db, array, subsite, node, inst, dmethods)
    merged = pd.merge(output_df, dbf, on='reference_designator', how='outer')

    url_list = build_data_request_urls(merged)

    urls = pd.DataFrame(url_list)

    dpath = os.path.join(sDir, 'data_request_urls_{}.csv'.format(now))
    if os.path.isfile(dpath):
        with open(dpath, 'a') as dd:
            writer = csv.writer(dd)
            for u in url_list:
                writer.writerow([u])
    else:
        urls.to_csv(dpath, index=False, header=False)

    return url_list


if __name__ == '__main__':
    sDir = '/Users/lgarzio/Documents/OOI'
    array = []  # ['CP','CE']
    subsite = []  # ['GI03FLMA','GI03FLMB']
    node = []
    inst = []  # ['CTDMO,FLOR']
    delivery_methods = []  # ['streamed','telemetered,'recovered']
    now = dt.datetime.now().strftime('%Y%m%dT%H%M')
    main(sDir, array, subsite, node, inst, delivery_methods, now)
