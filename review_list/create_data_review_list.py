#!/usr/bin/env python
"""
Created on Sep 4 2018

@author: Lori Garzio
@brief: This script creates a list of OOI 1.0 data sets for review
@usage:
rootdir: path to local asset management deployment repo
"""

import pandas as pd
import os

rootdir = '/Users/lgarzio/Documents/repo/lgarzio/ooi-integration-fork/asset-management/deployment'

db_inst_stream = pd.read_csv('https://raw.githubusercontent.com/seagrinch/data-team-python/master/infrastructure/data_streams.csv')
db_stream_desc = pd.read_csv('https://raw.githubusercontent.com/seagrinch/data-team-python/master/infrastructure/stream_descriptions.csv')

db_inst_stream = db_inst_stream[['reference_designator', 'method', 'stream_name']]
db_stream_desc = db_stream_desc.rename(columns={'name': 'stream_name'})
db_stream_desc = db_stream_desc[['stream_name', 'stream_type']]
db = pd.merge(db_inst_stream, db_stream_desc, on='stream_name', how='outer')
db = db[db.reference_designator.notnull()]
db = db[db.stream_type == 'Science']  # filter on science streams only
db = db[['reference_designator', 'method', 'stream_name']]

# Create a dataframe of just unique reference designators from the data team database
db_refdes = pd.DataFrame()
refdes_arr = db['reference_designator'].unique()
db_refdes['Reference Designator'] = refdes_arr
db_refdes['in_qcdb_science'] = 'yes'

am_df = pd.DataFrame()
for root, dirs, files in os.walk(rootdir):
    for f in files:
        if f.endswith('_Deploy.csv'):
            with open(os.path.join(root, f), 'r') as csv_file:
                df = pd.read_csv(csv_file)
                df = df[~df.CUID_Deploy.str.startswith('#')]  # remove deployment sheet lines commented out
                df = df[['Reference Designator', 'deploymentNumber', 'startDateTime', 'stopDateTime']]
                df['stopDateTime_dt'] = pd.to_datetime(df['stopDateTime'])
                am_df = am_df.append(df)

am_df = am_df.sort_values(by=['Reference Designator', 'deploymentNumber'])
am_df['in_am'] = 'yes'

ds = pd.merge(db_refdes, am_df, on='Reference Designator', how='outer')
ds['array_code'] = ds['Reference Designator'].str[0:2]
ds['subsite'] = ds['Reference Designator'].str.split('-').str[0]
ds['node'] = ds['Reference Designator'].str.split('-').str[1]
ds['sensor'] = ds['Reference Designator'].str.split('-').str[2] + '-' + ds['Reference Designator'].str.split('-').str[3]
ds = ds.sort_values(by=['subsite', 'node', 'sensor', 'deploymentNumber'])

ds['status'] = ''
ds.loc[((ds['in_am'] == 'yes') & (ds['in_qcdb_science'] == 'yes')), 'status'] = 'for review'
ds.loc[((ds['in_am'] == 'yes') & (ds['in_qcdb_science'].isnull())), 'status'] = 'not for review'
ds.loc[((ds['in_am'].isnull()) & (ds['in_qcdb_science'] == 'yes')), 'status'] = 'for review'
ds.loc[(ds['stopDateTime_dt'] >= '2018-10-01'), 'status'] = 'not for review'  # identify deployments that end before 2018-10-01
ds.loc[ds['stopDateTime'].isnull(), 'status'] = 'not for review'
# exclude instruments that are not processed through uFrame
ds.loc[(ds['sensor'].str.contains('ZPLS|CAMDS|FLOBN|OSMOIA|MASSP|PPSDNA|RASFL')), 'status'] = 'not for review'
ds.loc[(ds['sensor'].str.contains('BOTPT')), 'status'] = 'for review'  # include BOTPT (has no end date in AM)

cols = ['array_code', 'Reference Designator', 'subsite', 'node', 'sensor', 'deploymentNumber', 'startDateTime', 'stopDateTime', 'in_am',
        'in_qcdb_science', 'status']
ds.to_csv('data_review_list.csv', index=False, columns=cols)
