#!/usr/bin/env python
"""
Created on Dec 18 2018 by Lori Garzio
@brief This is a wrapper script that imports tools to calculate final statistics for science variables in multiple datasets.
@usage
sDir: location to save summary output
f: file containing THREDDs urls with .nc files to analyze. The column containing the THREDDs urls must be labeled
'outputUrl' (e.g. an output from one of the data_download scripts)
"""

import pandas as pd
import scripts

sDir = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/data_ranges'
plotting_sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
f = '/Users/lgarzio/Documents/OOI/DataReviews/GI/GI03FLMA/data_request_summary_run1.csv'
sd_calc = 12  # number of standard deviations for outlier calculation. options: int or None

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()
url_list = [u for u in url_list if u not in 'no_output_url']

scripts.final_ds_stats.main(sDir, plotting_sDir, url_list, sd_calc)
