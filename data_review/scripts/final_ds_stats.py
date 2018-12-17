#!/usr/bin/env python
"""
Created on 11/12/2018
@author Lori Garzio
@brief Calculate final statistics for science variables in a dataset. Long Names and units for science variables must
be the same among methods.
sDir: location to save summary output
url: list of THREDDs urls containing .nc files to analyze.
"""

import os
import itertools
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
import functions.common as cf
import functions.combine_datasets as cd


def format_dates(dd):
    fd = dt.datetime.strptime(dd.replace(',', ''), '%m/%d/%y %I:%M %p')
    fd2 = dt.datetime.strftime(fd, '%Y-%m-%dT%H:%M:%S')
    return fd2


def main(sDir, url_list):
    dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
    drn = dr.loc[dr.type == 'exclusion']
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        fdatasets = []
        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)
        pms = []
        for index, row in ps_df.iterrows():
            for ii in range(n_streams):
                rms = '-'.join((r, row[ii]))
                pms.append(row[ii])
                for dd in datasets:
                    spl = dd.split('/')[-2].split('-')
                    catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                    fdeploy = dd.split('/')[-1].split('_')[0]
                    if rms == catalog_rms and fdeploy == row['deployment']:
                        fdatasets.append(dd)

        main_sensor = r.split('-')[-1]
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        # find time ranges to exclude from analysis for data review database
        subsite = r.split('-')[0]
        subsite_node = '-'.join((subsite, r.split('-')[1]))

        drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]
        et = []
        for i, row in drne.iterrows():
            sdate = format_dates(row.start_date)
            edate = format_dates(row.end_date)
            et.append([sdate, edate])

        # get science variable long names from the Data Review Database
        stream_sci_vars = cd.sci_var_long_names(r)

        # check if the science variable long names are the same for each stream
        sci_vars_dict = cd.sci_var_long_names_check(stream_sci_vars)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        sci_vars_dict = cd.append_science_data(ps_df, n_streams, r, fdatasets_sel, sci_vars_dict, et)

        # analyze combined dataset
        print('\nAnalyzing combined dataset and writing summary file')

        array = subsite[0:2]
        save_dir = os.path.join(sDir, array, subsite)
        cf.create_dir(save_dir)

        headers = ['common_stream_name', 'preferred_methods_streams', 'long_name', 'units', 't0', 't1', 'fill_value',
                   'global_ranges', 'n_all', 'n_nans', 'n_fillvalues', 'n_grange', 'define_stdev', 'n_outliers',
                   'n_stats', 'mean', 'min', 'max', 'stdev']
        rows = []
        for m, n in sci_vars_dict.items():
            print(m)
            if m == 'common_stream_placeholder':
                m = 'science_data_stream'
            for sv, vinfo in n['vars'].items():
                print(sv)
                lst_fill_value = np.unique(vinfo['fv']).tolist()
                if len(lst_fill_value) == 1:
                    fill_value = lst_fill_value[0]
                else:
                    print('No unique fill value for {}'.format(sv))

                lunits = np.unique(vinfo['units']).tolist()

                t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                data = vinfo['values']
                n_all = len(data)

                # reject NaNs
                data_nonan = data[~np.isnan(data)]
                n_nan = n_all - len(data_nonan)

                # reject fill values
                data_nonan_nofv = data_nonan[data_nonan != fill_value]
                n_fv = n_all - n_nan - len(data_nonan_nofv)

                # reject data outside of global ranges
                ds0 = xr.open_dataset(fdatasets[0])
                for vv in list(ds0.data_vars.keys()):
                    try:
                        vv_ln = ds0[vv].long_name
                        vv_units = ds0[vv].units
                        if vv_ln == sv and vv_units[0]:
                            [g_min, g_max] = cf.get_global_ranges(r, vv)
                            break
                    except AttributeError:
                        continue

                if g_min is not None and g_max is not None:
                    gr_ind = cf.reject_global_ranges(data_nonan_nofv, g_min, g_max)
                    data_nonan_nofv_gr = data_nonan_nofv[gr_ind]
                    n_grange = n_all - n_nan - n_fv - len(data_nonan_nofv_gr)
                else:
                    n_grange = 'no global ranges'
                    data_nonan_nofv_gr = data_nonan_nofv

                if len(data_nonan_nofv_gr) > 1:
                    sd_calc = None  # number of standard deviations for outlier calculation. options: int or None
                    [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_nonan_nofv_gr, sd_calc)
                else:
                    sd_calc = None
                    num_outliers = None
                    mean = None
                    vmin = None
                    vmax = None
                    sd = None
                    n_stats = None

                rows.append([m, list(np.unique(pms)), sv, lunits, t0, t1, lst_fill_value, [g_min, g_max], n_all, n_nan,
                             n_fv, n_grange, sd_calc, num_outliers, n_stats, mean, vmin, vmax, sd])

        fsum = pd.DataFrame(rows, columns=headers)
        fsum.to_csv('{}/{}_final_stats.csv'.format(save_dir, r), index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/final_stats'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022407-GI03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022421-GI03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181127T022434-GI03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']

    main(sDir, url_list)
