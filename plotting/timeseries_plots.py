#!/usr/bin/env python
"""
Created on Dec 6 2018 by Lori Garzio
@brief This is a wrapper script that imports tools to plot a variety of timeseries plots for an instrument/deployment.
@usage
sDir: location to save summary output
f: file containing THREDDs urls with .nc files to analyze. The column containing the THREDDs urls must be labeled
'outputUrl' (e.g. an output from one of the data_download scripts)
start_time: optional start time to limit plotting time range
end_time: optional end time to limit plotting time range
preferred_only: if set to 'yes', only plots the preferred data for a deployment. Options are 'yes' or 'no'
"""

import pandas as pd
import datetime as dt
import scripts

sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
f = '/Users/lgarzio/Documents/OOI/DataReviews/GA/GA03FLMA/data_request_summary_run1.csv'
start_time = None  # dt.datetime(2015, 5, 16, 20, 0, 0)  # optional, set to None if plotting all data
end_time = None  # dt.datetime(2015, 5, 18, 0, 0, 0)  # optional, set to None if plotting all data
preferred_only = 'yes'  # options: 'yes', 'no'

ff = pd.read_csv(f)
url_list = ff['outputUrl'].tolist()
url_list = [u for u in url_list if u not in 'no_output_url']

scripts.plot_timeseries.main(sDir, url_list, start_time, end_time, preferred_only)
scripts.plot_timeseries_panel.main(sDir, url_list, start_time, end_time, preferred_only)
scripts.plot_timeseries_pm_all.main(sDir, url_list, start_time, end_time)
scripts.plot_timeseries_all.main(sDir, url_list)
scripts.plot_compare_timeseries.main(sDir, url_list, start_time, end_time)
scripts.plot_ts.main(sDir, url_list, start_time, end_time, preferred_only)
scripts.plot_timeseries_monthly.main(sDir, url_list)
scripts.plot_timeseries_daily.main(sDir, url_list)
