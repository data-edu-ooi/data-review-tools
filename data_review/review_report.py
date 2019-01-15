#!/usr/bin/env python
"""
Created on Dec 17 2018 by Lori Garzio
@brief This script creates a summary output of review notes from the Data Review Database, filtered by date
last modified
@usage
sDir: location to save summary output, and where the script finds the last time the summary files were run
"""

import pandas as pd
import datetime as dt
import os
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/review_reports'


def format_dates(dd):
    if dd == '':
        fd2 = ''
    else:
        fd = dt.datetime.strptime(dd.replace(',', ''), '%m/%d/%y %I:%M %p')
        fd2 = dt.datetime.strftime(fd, '%Y-%m-%dT%H:%M:%S')
    return fd2


def format_df(dataframe):
    for i, row in dataframe.iterrows():
        sdate = format_dates(row.start_date)
        edate = format_dates(row.end_date)
        mod_date = dt.datetime.strftime(row.modified, '%Y-%m-%dT%H:%M:%S')
        dataframe.loc[row.name, 'start_date'] = sdate
        dataframe.loc[row.name, 'end_date'] = edate
        dataframe.loc[row.name, 'modified'] = mod_date
    return dataframe


mdates = []
for root, dirs, files in os.walk(sDir):
    for f in files:
        if f.endswith('.csv'):
            md = f.split('_')[-1].split('.')[0]
            mdates.append(dt.datetime.strptime(md, '%Y%m%dT%H%M'))

mdate = max(mdates)

f = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
f['modified'] = f['modified'].map(lambda t: dt.datetime.strptime(t.replace(',', ''), '%m/%d/%y %I:%M %p'))
df = f.loc[(f['modified'] > mdate) & (f['type'] != 'draft')]
df = df.fillna('')

# write file containing all notes that were modified since the last output
format_df(df)
sfile = 'OOI_datareview_report_{}.csv'.format(dt.datetime.utcnow().strftime('%Y%m%dT%H%M'))
df.to_csv(os.path.join(sDir, sfile), index=False)

# write file containing all of the draft notes
df_drafts = f.loc[f['type'] == 'draft']
df_drafts = df_drafts.fillna('')
if not df_drafts.empty:
    format_df(df_drafts)
    sfile_drafts = 'OOI_datareview_drafts_{}.csv'.format(dt.datetime.utcnow().strftime('%Y%m%dT%H%M'))
    df_drafts.to_csv(os.path.join(sDir, sfile_drafts), index=False)
