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
import functions.common as cf
from tools import nc_file_analysis, define_preferred_stream, compare_methods, nc_file_summary

sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output'
f = '/Users/lgarzio/Documents/OOI/DataReviews/test/data_request_summary.csv'

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()

# check that the requests have fulfilled before continuing with the analysis
print 'Seeing if the requests have fulfilled...'
for i in range(len(url_list)):
    url = url_list[i]
    print '\nDataset {} of {}: {}'.format((i + 1), len(url_list), url)
    cf.check_request_status(url)

json_nc_analysis = nc_file_analysis.main(sDir, url_list)

json_method_comparison = compare_methods.main(sDir, url_list)

for j in json_nc_analysis:
    refdes = j.split('/')[-2]
    ps = define_preferred_stream.main(j)
    mc = [k for k in json_method_comparison if refdes in k]
    if len(mc) == 1:
        print '{}: writing summary files'.format(refdes)
        nc_file_summary.main(j, ps, mc[0])
    else:
        print 'Too many method comparison files provided.'
