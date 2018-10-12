#!/usr/bin/env python
"""
Created on Feb 2 2017 by Mike Smith
Modified on Oct 11 2018 by Lori Garzio
@brief This is a wrapper script that imports the tools: nc_file_analysis and nc_file_summary as python methods.
@usage
sDir: location to save summary output
f: file containing THREDDs urls with .nc files to analyze. The column containing the THREDDs urls must be labeled
'outputUrl' (e.g. an output from one of the data_download scripts)
"""

import pandas as pd

from tools import nc_file_analysis, nc_file_summary

sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output'
f = '/Users/lgarzio/Documents/OOI/DataReviews/test/data_request_summary.csv'

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()

json_files = nc_file_analysis.main(sDir, url_list)

for j in json_files:
    nc_file_summary.main(j)
