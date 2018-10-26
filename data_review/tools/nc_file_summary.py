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
import ast


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
    loc_compare = data['location_comparison']
    deployments = np.sort(data['deployments'].keys()).tolist()

    if type(ps) == str:
        ps = load_json_file(ps)

    if type(mc) == str:
        mc = load_json_file(mc)

    fsummary_headers = ['deployment', 'file_downloaded', 'preferred_method', 'stream', 'other_methods',
                        'time_delta_start', 'time_delta_end', 'start_days_missing', 'end_days_missing',
                        'location_diff_km', 'n_days_deployed', 'n_timestamps', 'n_days', 'deploy_depth', 'pressure_mean',
                        'pressure_diff', 'pressure_var', 'pressure_units', 'num_pressure_outliers', 'missing_vars_file',
                        'missing_vars_db', 'file_time_gaps', 'gaps_num', 'gaps_num_days',
                        'timestamp_test', 'valid_data_test', 'full_dataset_test', 'file_coordinates', 'coordinate_test', 'filename']
    vsummary_headers = ['deployment', 'preferred_method', 'stream', 'variable', 'units', 'fill_value',
                        'n_all', 'n_outliers', 'n_nans', 'n_fillvalues', 'n_stats', 'percent_valid_data', 'mean', 'min', 'max', 'stdev']
    csummary_headers = ['deployment', 'preferred_method_stream', 'comparison_method_stream', 'long_name',
                        'preferred_name', 'preferred_units', 'unit_comparison_test', 'preferred_n', 'preferred_n_nan',
                        'missing_data', 'n_comparison', 'min_abs_diff', 'max_abs_diff', 'n_diff_greater_zero']
    fsummary_rows = []
    vsummary_rows = []
    csummary_rows = []
    for dindex, d in enumerate(deployments):
        ddata = data['deployments'][d]
        start = ddata['deploy_start']
        stop = ddata['deploy_stop']
        depth = ddata['deploy_depth']
        n_days_deployed = ddata['n_days_deployed']
        for m in ddata['method'].keys():
            for s in ddata['method'][m]['stream'].keys():
                ms = '-'.join((m, s))
                if ms in ps[d]:
                    # build the summary of comparison of science variables among delivery methods
                    missing_data_list = []
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

                                    missing_data_list.append(str(missing_data))

                                    csummary_rows.append([d, str(preferred_stream_name), str(comparison_stream_name),
                                                          var, name, units, unit_test, n, n_nan, missing_data,
                                                          n_comparison, min_abs_diff, max_abs_diff,
                                                          n_diff_greater_zero])
                    except KeyError:
                        csummary_rows.append([d, ms, 'no other methods available for comparison', None, None, None,
                                              None, None, None, None, None, None, None, None])

                    for fname in ddata['method'][m]['stream'][s]['file'].keys():
                        fsummary = ddata['method'][m]['stream'][s]['file'][fname]
                        nt = fsummary['n_timestamps']

                        # build the summary of science variable statistics
                        valid_list = []
                        for v in ddata['method'][m]['stream'][s]['file'][fname]['sci_var_stats'].keys():
                            vsummary = data['deployments'][d]['method'][m]['stream'][s]['file'][fname]['sci_var_stats'][v]
                            units = vsummary['units']
                            mean = vsummary['mean']
                            min = vsummary['min']
                            max = vsummary['max']
                            stdev = vsummary['stdev']
                            n_stats = vsummary['n_stats']
                            percent_valid_data = round((float(n_stats)/float(nt) * 100), 2)
                            n_o = vsummary['n_outliers']
                            n_nan = vsummary['n_nans']
                            n_fv = vsummary['n_fillvalues']
                            fv = vsummary['fill_value']

                            valid_list.append(percent_valid_data)

                            vsummary_rows.append([d, m, s, v, units, fv, nt, n_o, n_nan, n_fv, n_stats, percent_valid_data, mean, min,
                                                  max, stdev])

                        # build file summary
                        other_methods = [str(x) for x in ddata['method'].keys() if x not in [m]]
                        dwnl = fsummary['file_downloaded']
                        coords = fsummary['file_coordinates']
                        dstart = fsummary['data_start']
                        dstop = fsummary['data_stop']
                        gaps = fsummary['time_gaps']
                        n_gaps = len(gaps)
                        v_missing_f = fsummary['vars_not_in_file']
                        v_missing_db = fsummary['vars_not_in_db']
                        nd = fsummary['n_days']
                        vpress = fsummary['pressure_comparison']['variable']
                        mpress = fsummary['pressure_comparison']['pressure_mean']
                        upress = fsummary['pressure_comparison']['units']
                        press_diff = fsummary['pressure_comparison']['diff']
                        opress = fsummary['pressure_comparison']['num_outliers']

                        tdelta_start = time_delta(start, dstart)
                        if stop == 'None':
                            tdelta_stop = 'no end date in AM'
                            tdelta_stop_days = None
                        else:
                            tdelta_stop = time_delta(dstop, stop)
                            tdelta_stop_days = tdelta_stop.days

                        # Summarize the timestamps tests
                        at = fsummary['ascending_timestamps']
                        ut = fsummary['unique_timestamps']
                        if 'pass' in str(at) and 'pass' in str(ut):
                            time_test = 'pass'
                        elif 'pass' in str(at) and 'fail' in str(ut):
                            time_test = 'fail unique timestamp test'
                        elif 'fail' in str(at) and 'pass' in str(ut):
                            time_test = 'fail ascending timestamp test'
                        elif 'fail' in str(at) and 'fail' in str(ut):
                            time_test = 'fail unique and ascending timestamp tests'

                        # Location difference from first deployment of instrument
                        loc_diff = []
                        if dindex == 0:
                            loc_diff = []
                        else:
                            d1 = ''.join(('D', str(d[-1])))
                            for i in range(dindex):
                                d0 = ''.join(('D', str(deployments[i][-1])))
                                for k, value in loc_compare.iteritems():
                                    kk = k.split('_')
                                    if all(elem in kk for elem in [d1, d0]):
                                        loc_diff.append(value)

                        # Count total number of days missing within file
                        n_gaps_days = 0
                        for g in gaps:
                            add_days = (pd.to_datetime(g[1]) - pd.to_datetime(g[0])).days
                            n_gaps_days = n_gaps_days + add_days

                        # Check if any percent_valid_data values (for science variables) are < 95
                        pvd_test = dict()
                        x99 = len([x for x in valid_list if x > 99.00])
                        if x99 > 0:
                            pvd_test['99'] = x99

                        x95 = len([x for x in valid_list if 95.00 < x <= 99.00])
                        if x95 > 0:
                            pvd_test['95'] = x95

                        x75 = len([x for x in valid_list if 75.00 < x <= 95.00])
                        if x75 > 0:
                            pvd_test['75'] = x75

                        x50 = len([x for x in valid_list if 50.00 < x <= 75.00])
                        if x50 > 0:
                            pvd_test['50'] = x50

                        x25 = len([x for x in valid_list if 25.00 < x <= 50.00])
                        if x25 > 0:
                            pvd_test['25'] = x25

                        x0 = len([x for x in valid_list if x <= 25.00])
                        if x0 > 0:
                            pvd_test['0'] = x0

                        # Check if data are found in a "non-preferred" stream for any science variable
                        md_unique = np.unique(missing_data_list).tolist()
                        if len(md_unique) == 0:
                            fd_test = None
                        elif len(md_unique) == 1 and 'no missing data' in md_unique[0]:
                            fd_test = 'pass'
                        else:
                            n_missing_gaps = []
                            n_missing_days = []
                            for md in md_unique:
                                md_missing_days = 0
                                md = ast.literal_eval(md)
                                for md_gap in md['missing_data_gaps']:
                                    md_add_days = (pd.to_datetime(md_gap[1]) - pd.to_datetime(md_gap[0])).days
                                    md_missing_days = md_missing_days + md_add_days
                                n_missing_gaps.append(len(md['missing_data_gaps']))
                                n_missing_days.append(md_missing_days)
                            n_missing_gaps = np.unique([np.amin(n_missing_gaps), np.amax(n_missing_gaps)]).tolist()
                            n_missing_days = np.unique([np.amin(n_missing_days), np.amax(n_missing_days)]).tolist()

                            fd_test = 'fail: data found in another stream (gaps: {} days: {})'.format(n_missing_gaps,
                                                                                                      n_missing_days)

                        # Check the coordinates in the file
                        check_coords = list(set(['obs', 'time', 'pressure', 'lat', 'lon']) - set(coords))

                        if len(check_coords) > 0:
                            coord_test = 'missing coords: {}'.format(check_coords)
                        else:
                            coord_test = 'all expected coords listed'

                        fsummary_rows.append([d, dwnl, m, s, other_methods, str(tdelta_start), str(tdelta_stop),
                                              tdelta_start.days, tdelta_stop_days, loc_diff, n_days_deployed, nt,
                                              nd,
                                              depth, mpress, press_diff, vpress, upress, opress, v_missing_f,
                                              v_missing_db, gaps, n_gaps, n_gaps_days, time_test, pvd_test, fd_test, coords, coord_test, fname])

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
