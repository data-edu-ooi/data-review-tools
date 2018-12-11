#!/usr/bin/env python
"""
Created on Jan 11 2018
Modified on Sep 7 2018
@author: Lori Garzio
@brief: This is a collection of tools for use in creating data request urls.
"""

import itertools
import pandas as pd
import datetime as dt
import ast


def check_str(x):
    if type(x) == str:
        y = x
    else:
        y = []
    return y


def define_methods(delivery_method):
    valid_inputs = ['streamed', 'telemetered', 'recovered']
    dmethods = []
    if not delivery_method:
        dmethods = ['streamed', 'telemetered', 'recovered_host', 'recovered_inst', 'recovered_wfp', 'recovered_cspp', 'no_method']

    for d in delivery_method:
        if d not in valid_inputs:
            raise Exception('Invalid delivery_method: %s' %d)
        else:
            if d == 'recovered':
                dmethods.extend(['recovered_host', 'recovered_inst', 'recovered_wfp', 'recovered_cspp'])
            else:
                methods = d
                dmethods.append(methods)
    return dmethods


def filter_dataframe(df, array, subsite, node, inst, dmethods=''):
    df_filtered = pd.DataFrame()

    if not array:
        dba = df
    else:
        dba = df[df['array_code'].isin(array)]

    if not subsite:
        dbs = dba
    else:
        dbs = dba[dba['subsite'].isin(subsite)]

    if not node:
        dbn = dbs
    else:
        dbn = dbs[dbs['node'].isin(node)]

    if not dmethods:
        dbd = dbn
    else:
        dbd = dbn[dbn['method'].isin(dmethods)]

    if not inst:
        dbi = dbd
        df_filtered = dbi
    else:
        for i in inst:
            dbi = dbd.loc[dbd.sensor.str.contains(i)]
            if dbi.empty:
                continue
            df_filtered = df_filtered.append(dbi, ignore_index=True)

    return df_filtered


def format_date(date_text):
    if not date_text:
        fdate = ''
    else:
        try:
            dt.datetime.strptime(date_text, '%Y-%m-%dT%H:%M:%S')
            fdate = date_text + '.000Z'
        except ValueError:
            raise ValueError
    return fdate


def format_inputs(input_str):
    if input_str == '':
        formatted_input = []
    else:
        try:
            finput = ast.literal_eval(input_str)
        except (ValueError, SyntaxError):
            finput = input_str
        if type(finput) == list:
            formatted_input = finput
        elif type(finput) == tuple:
            formatted_input = list(finput)
        elif ',' in finput:
            finput = finput.replace(" ", "")  # remove any whitespace
            formatted_input = finput.split(',')
        else:
            formatted_input = [finput]

    return formatted_input


def filter_refdes(refdes_list, filter_by):
    alist = []
    for item in filter_by:
        filter = [x for x in refdes_list if item in x]
        alist.append(filter)
    flist = list(itertools.chain(*alist))
    db = pd.DataFrame(flist)
    return db


def get_database():
    db_inst_stream = pd.read_csv('https://raw.githubusercontent.com/seagrinch/data-team-python/master/infrastructure/data_streams.csv')
    db_stream_desc = pd.read_csv('https://raw.githubusercontent.com/seagrinch/data-team-python/master/infrastructure/stream_descriptions.csv')
    db_regions = pd.read_csv('https://raw.githubusercontent.com/seagrinch/data-team-python/master/infrastructure/regions.csv')

    db_inst_stream = db_inst_stream[['reference_designator', 'method', 'stream_name']]
    db_stream_desc = db_stream_desc.rename(columns={'name': 'stream_name'})
    db_stream_desc = db_stream_desc[['stream_name', 'stream_type']]
    db_merged = pd.merge(db_inst_stream, db_stream_desc, on='stream_name', how='outer')
    # drop duplicate rows (in the case where telemetered and recovered stream names are the same)
    db_merged = db_merged.drop_duplicates()
    db_merged['in_qcdb'] = 'yes'
    db_merged = db_merged[db_merged.reference_designator.notnull()]
    db_merged['array_code'] = db_merged['reference_designator'].str[0:2]
    db_regions = db_regions.rename(columns={'reference_designator': 'array_code', 'name': 'array_name'})
    db = pd.merge(db_merged, db_regions, on='array_code', how='outer')
    db = db.rename(columns={'name': 'stream_name'})

    db = db.fillna(value={'method': 'no_method'})

    db['subsite'] = db['reference_designator'].str.split('-').str[0]
    db['node'] = db['reference_designator'].str.split('-').str[1]
    db['sensor'] = db['reference_designator'].str.split('-').str[2] + '-' + db['reference_designator'].str.split('-').str[3]

    return db


def refdes_format(refdes):
    rd = refdes.split('-')
    inst_format = '{:s}/{:s}/{:s}-{:s}/'.format(rd[0], rd[1], rd[2], rd[3])
    return inst_format