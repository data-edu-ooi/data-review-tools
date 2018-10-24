#!/usr/bin/env python
"""
Created on 10/10/2018
@author Lori Garzio
@brief Provides human-readable .csv summaries of the json output from the nc_file_analysis tool: 1) file summary,
2) science variable comparison, and 3) science variable statistics. Saves the output in the same directory as the .json
files provided.
"""


import os
import pandas as pd
import numpy as np
import json


def load_json_file(f):
    if f.endswith('.json'):
        info = json.load(open(f, 'r'))
    else:
        print 'Not a .json file'
    return info


def time_delta(t0, t1):
    # calculate the time difference between 2 dates
    time_delta = pd.to_datetime(t1) - pd.to_datetime(t0)
    return time_delta


def main(f, ps, mc):
    data = load_json_file(f)
    refdes = data['refdes']
    deployments = np.sort(data['deployments'].keys()).tolist()

    if type(ps) == str:
        ps = load_json_file(ps)

    if type(mc) == str:
        mc = load_json_file(mc)

    fsummary_headers = ['deployment', 'file_downloaded', 'preferred_method', 'stream', 'other_methods',
                        'time_delta_start', 'time_delta_stop', 'n_days_deployed', 'n_timestamps', 'n_days', 'deploy_depth', 'pressure_mean',
                        'pressure_diff', 'pressure_var', 'pressure_units', 'num_pressure_outliers', 'missing_vars_file',
                        'missing_vars_db', 'file_time_gaps', 'ascending_timestamp_test', 'unique_timestamp_test',
                        'file_coordinates', 'filename']
    vsummary_headers = ['deployment', 'file_downloaded', 'preferred_method', 'stream', 'variable', 'units', 'fill_value',
                        'n_all', 'n_outliers', 'n_nans', 'n_fillvalues', 'n_stats', 'mean', 'min', 'max', 'stdev',
                        'filename']
    csummary_headers = ['deployment', 'preferred_method_stream', 'comparison_method_stream', 'long_name',
                        'preferred_name', 'preferred_units', 'unit_comparison_test', 'preferred_n', 'preferred_n_nan',
                        'missing_data', 'n_comparison', 'min_abs_diff', 'max_abs_diff', 'n_diff_greater_zero']
    fsummary_rows = []
    vsummary_rows = []
    csummary_rows = []
    for d in deployments:
        ddata = data['deployments'][d]
        start = ddata['deploy_start']
        stop = ddata['deploy_stop']
        depth = ddata['deploy_depth']
        n_days_deployed = ddata['n_days_deployed']
        for m in ddata['method'].keys():
            for s in ddata['method'][m]['stream'].keys():
                ms = '-'.join((m, s))
                if ms in ps[d]:
                    for fname in ddata['method'][m]['stream'][s]['file'].keys():
                        other_methods = [str(x) for x in ddata['method'].keys() if x not in [m]]
                        fsummary = ddata['method'][m]['stream'][s]['file'][fname]
                        dwnl = fsummary['file_downloaded']
                        coords = fsummary['file_coordinates']
                        dstart = fsummary['data_start']
                        dstop = fsummary['data_stop']
                        gaps = fsummary['time_gaps']
                        v_missing_f = fsummary['vars_not_in_file']
                        v_missing_db = fsummary['vars_not_in_db']
                        at = fsummary['ascending_timestamps']
                        ut = fsummary['unique_timestamps']
                        nt = fsummary['n_timestamps']
                        nd = fsummary['n_days']
                        vpress = fsummary['pressure_comparison']['variable']
                        mpress = fsummary['pressure_comparison']['pressure_mean']
                        upress = fsummary['pressure_comparison']['units']
                        press_diff = fsummary['pressure_comparison']['diff']
                        opress = fsummary['pressure_comparison']['num_outliers']

                        tdelta_start = time_delta(start, dstart)
                        if stop == 'None':
                            tdelta_stop = 'no end date in asset management'
                        else:
                            tdelta_stop = time_delta(dstop, stop)

                        fsummary_rows.append([d, dwnl, m, s, other_methods, str(tdelta_start), str(tdelta_stop), n_days_deployed, nt, nd,
                                              depth, mpress, press_diff, vpress, upress, opress, v_missing_f,
                                              v_missing_db, gaps, at, ut, coords, fname])

                        # build the summary of comparison of science variables among delivery methods
                        try:
                            for compare_str in mc['deployments'][d]['comparison'].keys():
                                if str(ms) in str(compare_str):
                                    [ds0, ds1] = [compare_str.split(' ')[0], compare_str.split(' ')[1]]
                                    if ms == ds0:
                                        preferred_stream = 'ds0'
                                        preferred_stream_name = ds0
                                        comparison_stream_name = ds1
                                    else:
                                        preferred_stream = 'ds1'
                                        preferred_stream_name = ds1
                                        comparison_stream_name = ds0

                                    for var in mc['deployments'][d]['comparison'][compare_str]['vars'].keys():
                                        compare_summary = mc['deployments'][d]['comparison'][compare_str]['vars'][var]
                                        name = compare_summary[preferred_stream]['name']
                                        units = compare_summary[preferred_stream]['units']
                                        unit_test = compare_summary['unit_test']
                                        n = compare_summary[preferred_stream]['n']
                                        n_nan = compare_summary[preferred_stream]['n_nan']
                                        missing_data = compare_summary[preferred_stream]['missing']
                                        n_comparison = compare_summary['n_comparison']
                                        min_abs_diff = compare_summary['min_abs_diff']
                                        max_abs_diff = compare_summary['max_abs_diff']
                                        n_diff_greater_zero = compare_summary['n_diff_greater_zero']

                                        csummary_rows.append([d, str(preferred_stream_name), str(comparison_stream_name),
                                                              var, name, units, unit_test, n, n_nan, missing_data,
                                                              n_comparison, min_abs_diff, max_abs_diff,
                                                              n_diff_greater_zero])
                        except KeyError:
                            csummary_rows.append([d, ms, 'no other methods available for comparison', None, None, None,
                                                  None, None, None, None, None, None, None, None])

                        # build the summary of science variable statistics
                        for v in ddata['method'][m]['stream'][s]['file'][fname]['sci_var_stats'].keys():
                            vsummary = data['deployments'][d]['method'][m]['stream'][s]['file'][fname]['sci_var_stats'][v]
                            units = vsummary['units']
                            mean = vsummary['mean']
                            min = vsummary['min']
                            max = vsummary['max']
                            stdev = vsummary['stdev']
                            n_stats = vsummary['n_stats']
                            n_o = vsummary['n_outliers']
                            n_nan = vsummary['n_nans']
                            n_fv = vsummary['n_fillvalues']
                            fv = vsummary['fill_value']

                            vsummary_rows.append([d, dwnl, m, s, v, units, fv, nt, n_o, n_nan, n_fv, n_stats, mean, min,
                                                  max, stdev, fname])

    fdf = pd.DataFrame(fsummary_rows, columns=fsummary_headers)
    fdf.to_csv('{}/{}_file_summary.csv'.format(os.path.dirname(f), refdes), index=False)

    cdf = pd.DataFrame(csummary_rows, columns=csummary_headers)
    cdf.to_csv('{}/{}_sciencevar_comparison_summary.csv'.format(os.path.dirname(f), refdes), index=False, encoding='utf-8')

    vdf = pd.DataFrame(vsummary_rows, columns=vsummary_headers)
    vdf.to_csv('{}/{}_sciencevar_summary.csv'.format(os.path.dirname(f), refdes), index=False, encoding='utf-8')


if __name__ == '__main__':
    f = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output/GP03FLMA/GP03FLMA-RIM01-02-CTDMOG040/GP03FLMA-RIM01-02-CTDMOG040-file_analysis.json'
    ps = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output/GP03FLMA/GP03FLMA-RIM01-02-CTDMOG040/GP03FLMA-RIM01-02-CTDMOG040-preferred_stream.json'
    mc = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output/GP03FLMA/GP03FLMA-RIM01-02-CTDMOG040/GP03FLMA-RIM01-02-CTDMOG040-method_comparison.json'

    main(f, ps, mc)
