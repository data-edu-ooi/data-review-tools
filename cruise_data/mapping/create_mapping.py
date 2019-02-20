#!/usr/bin/env python
"""
Created on Feb 19 2019

@author: Lori Garzio
@brief: create a mapping of OOI platform deployments and cruise CTD files from previous documents
"""

import pandas as pd
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console

mfile = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/cruise_data/old_files/platform_CTDcast_mapping.csv'
cfile = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/cruise_data/old_files/cruise_CTDs.csv'
wsfile = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/cruise_data/old_files/cruise_watersampling.csv'

mapping = pd.read_csv(mfile).fillna('')
ctds = pd.read_csv(cfile, encoding='ISO-8859-1').fillna('')
ws = pd.read_csv(wsfile)
ws.dropna(subset=['CTD_CruiseName'], inplace=True)  # get rid of the small Endurance cruises
ws.drop(columns=['Notes', 'oxygen_filename'], inplace=True)
ws = ws.fillna('')
ws['filepath_primary'] = ws['filepath_primary'].map(lambda x: x[16:])
check_columns = ['CTDlog', 'Chla', 'Nutrients', 'DIC_TA_pH']
for c in check_columns:
    ws[c] = ws['filepath_water_sampling'] + ws[c].map(str)

headers = ['array', 'platform', 'deploymentNumber', 'deployment', 'CTD_CruiseName', 'CTD_CruiseLeg', 'CTD_cast',
           'CTD_Date', 'CTD_lat', 'CTD_lon', 'alfresco_location_primary', 'CTD_cast_file']
summary = []

for i, r in mapping.iterrows():
    if type(r.CTDcast) == float:
        ff = ctds.loc[(ctds['CTD_CruiseName'] == r['CTD_CruiseName']) & (ctds['CTD_CruiseLeg'] == r['CTD_CruiseLeg'])]
        for ii, row in ff.iterrows():
            try:
                if int(row['CTDcast']) == int(r['CTDcast']):
                    filepath = row.filepath_primary[16:]
                    fpath = '/'.join(filepath.split('/')[:-2])
                    if len(fpath) > 0:
                        fpath = fpath + '/'
                    ctd_file = row.CTD_rawdata_filepath
                    if not ctd_file.endswith(('.cnv', '.hex')):
                        ctd_file = 'missing'
                    else:
                        ctd_file = '/'.join((filepath.split('/')[-2], ctd_file))
                    summary.append([r.Array, r.platform, r.deploymentNumber, r.Deployment, r.CTD_CruiseName,
                                    r.CTD_CruiseLeg, r.CTDcast, row.CTD_Date, row.CTD_lat, row.CTD_lon,
                                    fpath, ctd_file])
            except ValueError:
                continue
    else:
        summary.append([r.Array, r.platform, r.deploymentNumber, r.Deployment, r.CTD_CruiseName, r.CTD_CruiseLeg,
                        '', '', '', '', '', ''])

df = pd.DataFrame(summary, columns=headers)
df2 = pd.merge(df, ws, on=['array', 'CTD_CruiseName', 'CTD_CruiseLeg'], how='outer')
df2.sort_values(by=['platform', 'deploymentNumber'], inplace=True)
df2 = df2.fillna('')

fsheaders = ['array', 'platform', 'deploymentNumber', 'deployment', 'CTD_CruiseName', 'CTD_CruiseLeg', 'CTD_cast',
             'CTD_Date', 'CTD_lat', 'CTD_lon', 'alfresco_location_primary', 'Quick_Look', 'Cruise_Report',
             'CTD_cast_file', 'CTDlog', 'Chla', 'Nutrients', 'DIC_TA_pH']
final_summary = []
for i, rr in df2.iterrows():
    if len(rr.alfresco_location_primary) > 0:
        alp = rr.alfresco_location_primary
    else:
        alp = rr.filepath_primary
    for cc in check_columns:
        if 'missing' in rr[cc]:
            rr[cc] = 'missing'

    final_summary.append([rr.array, rr.platform, rr.deploymentNumber, rr.deployment, rr.CTD_CruiseName, rr.CTD_CruiseLeg,
                          rr.CTD_cast, rr.CTD_Date, rr.CTD_lat, rr.CTD_lon, alp, rr.Quick_Look_Report, rr.Cruise_Report,
                          rr.CTD_cast_file, rr.CTDlog, rr.Chla, rr.Nutrients, rr.DIC_TA_pH])

fs = pd.DataFrame(final_summary, columns=fsheaders)
fs.to_csv('/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/cruise_data/platform_cruisedata_mapping.csv', index=False)
