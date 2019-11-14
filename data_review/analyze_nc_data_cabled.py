#!/usr/bin/env python
"""
Created on Feb 2 2017 by Mike Smith
Modified on Oct 11 2018 by Lori Garzio
@brief This is a wrapper script that imports tools to analyze cabled OOI netCDF files and provide summary outputs.
@usage
sDir: location to save summary output
f: file containing THREDDs urls with .nc files to analyze. The column containing the THREDDs urls must be labeled
'outputUrl' (e.g. an output from one of the data_download scripts)
"""

import pandas as pd
import scripts

sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/output'
f = '/Users/lgarzio/Documents/OOI/DataReviews/CE/CE04OSPS/data_request_summary_ctd.csv'
# to run in server
sDir = '/home/lgarzio/repo/OOI/ooi-data-lab/data-review-tools/data_review/output'
f = '/home/lgarzio/OOI/DataReviews/CE/CE04OSPS/data_request_summary_ctd.csv'

deployment_num = 1  # None or int

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()
url_list = [u for u in url_list if (u not in 'no_output_url') & ('ENG' not in str(u))]

json_nc_analysis = scripts.nc_file_analysis_cabled.main(sDir, url_list, deployment_num)

# for j in json_nc_analysis:
#     refdes = j.split('/')[-2]
#     ps = scripts.define_preferred_stream.main(j)
#     mc = [k for k in json_method_comparison if refdes in k]
#     if len(mc) == 1:
#         print('{}: writing summary files'.format(refdes))
#         scripts.nc_file_summary.main(j, ps, mc[0])
#     elif len(mc) == 0:
#         print('No method comparison files provided.')
#     else:
#         print('Too many method comparison files provided.')
