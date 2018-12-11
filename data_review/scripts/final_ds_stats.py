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
import ast
import functions.common as cf
import functions.combine_datasets as cd


def main(sDir, url_list):
    eliminate_times = pd.read_csv('/Users/lgarzio/Documents/OOI/DataReviews/test4/eliminate_times.csv')
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
        for index, row in ps_df.iterrows():
            for ii in range(n_streams):
                rms = '-'.join((r, row[ii]))
                for dd in datasets:
                    spl = dd.split('/')[-2].split('-')
                    catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                    fdeploy = dd.split('/')[-1].split('_')[0]
                    if rms == catalog_rms and fdeploy == row['deployment']:
                        fdatasets.append(dd)

        main_sensor = r.split('-')[-1]
        fdatasets_sel = cf.filter_collocated_instruments(main_sensor, fdatasets)

        et = ast.literal_eval(eliminate_times.loc[eliminate_times['refdes'] == r]['times'][1])

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

        subsite = r.split('-')[0]
        array = subsite[0:2]
        save_dir = os.path.join(sDir, array, subsite, r)
        cf.create_dir(save_dir)

        headers = ['common_stream_name', 'potential_methods_streams', 'long_name', 'units', 't0', 't1', 'fill_value',
                   'n_all', 'n_outliers', 'n_nans', 'n_fillvalues', 'n_stats', 'mean', 'min', 'max', 'stdev']
        rows = []
        for m, n in sci_vars_dict.items():
            print(m)
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

                if len(data_nonan_nofv) > 1:
                    [num_outliers, mean, vmin, vmax, sd, n_stats] = cf.variable_statistics(data_nonan_nofv, 3)
                else:
                    num_outliers = None
                    mean = None
                    vmin = None
                    vmax = None
                    sd = None
                    n_stats = None

                rows.append([m, n['ms'], sv, lunits, t0, t1, lst_fill_value, n_all, num_outliers, n_nan,
                             n_fv, n_stats, mean, vmin, vmax, sd])

        fsum = pd.DataFrame(rows, columns=headers)
        fsum.to_csv('{}/{}_final_stats.csv'.format(save_dir, r), index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/lgarzio/Documents/OOI/DataReviews'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172034-GP03FLMA-RIM01-02-CTDMOG040-recovered_inst-ctdmo_ghqr_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172050-GP03FLMA-RIM01-02-CTDMOG040-recovered_host-ctdmo_ghqr_sio_mule_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181128T172104-GP03FLMA-RIM01-02-CTDMOG040-telemetered-ctdmo_ghqr_sio_mule_instrument/catalog.html']

    main(sDir, url_list)
