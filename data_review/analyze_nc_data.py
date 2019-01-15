#!/usr/bin/env python
"""
Created on Feb 2 2017 by Mike Smith
Modified on Oct 11 2018 by Lori Garzio
@brief This is a wrapper script that imports tools to analyze OOI netCDF files and provide summary outputs.
@usage
sDir: location to save summary output
f: file containing THREDDs urls with .nc files to analyze. The column containing the THREDDs urls must be labeled
'outputUrl' (e.g. an output from one of the data_download scripts)
"""

import pandas as pd
import scripts

sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/output'
f = '/Users/lgarzio/Documents/OOI/DataReviews/test3/GS01SUMO/data_request_summary_20181026T0933.csv'
#sDir = '/Users/lgarzio/Documents/OOI/DataReviews/test3'
#f = '/Users/lgarzio/Documents/OOI/DataReviews/test/data_request_summary_metbk.csv'

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()
url_list = [u for u in url_list if u not in 'no_output_url']

json_nc_analysis = scripts.nc_file_analysis.main(sDir, url_list)

json_method_comparison = scripts.compare_methods.main(sDir, url_list)

for j in json_nc_analysis:
    refdes = j.split('/')[-2]
    ps = scripts.define_preferred_stream.main(j)
    mc = [k for k in json_method_comparison if refdes in k]
    if len(mc) == 1:
        print('{}: writing summary files'.format(refdes))
        scripts.nc_file_summary.main(j, ps, mc[0])
    elif len(mc) == 0:
        print('No method comparison files provided.')
    else:
        print('Too many method comparison files provided.')
