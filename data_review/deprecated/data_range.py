#!/usr/bin/env python
"""
Created on March 2019

@author: Leila Belabbassi
@brief: This script is specific to instruments data on mobile platforms (WFP & Gliders).
It is used to generate:
 a csv file with data ranges calculated for a-user selected depth cell size (e.g., cell-size = 10 dbar)
 a plot with all deployments data filtered out of  erroneous and suspect data.
"""

import datetime as dt
import functions.plotting as pf
import functions.common as cf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import itertools
import numpy as np
import xarray as xr
import datetime


def main(url_list, sDir, mDir, zcell_size, zdbar, start_time, end_time):
    """""
    URL : path to instrument data by methods
    sDir : path to the directory on your machine to save files
    plot_type: folder name for a plot type

    """""
    rd_list = []
    ms_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = uu.split(rd + '-')[1].split('/')[0]
        if rd not in rd_list:
            rd_list.append(rd)
        if ms not in ms_list:
            ms_list.append(ms)

    ''' 
    separate different instruments
    '''
    for r in rd_list:
        print('\n{}'.format(r))
        subsite = r.split('-')[0]
        array = subsite[0:2]
        main_sensor = r.split('-')[-1]

        # read in the analysis file
        dr_data = cf.refdes_datareview_json(r)

        # get the preferred stream information
        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # get science variable long names from the Data Review Database
        stream_sci_vars = cd.sci_var_long_names(r)
        #stream_vars = cd.var_long_names(r)

        # check if the science variable long names are the same for each stream and initialize empty arrays
        sci_vars_dict0 = cd.sci_var_long_names_check(stream_sci_vars)

        # get the list of data files and filter out collocated instruments and other streams
        datasets = []
        for u in url_list:
            print(u)
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)

        datasets = list(itertools.chain(*datasets))
        fdatasets = cf.filter_collocated_instruments(main_sensor, datasets)
        fdatasets = cf.filter_other_streams(r, ms_list, fdatasets)

        # select the list of data files from the preferred dataset for each deployment
        fdatasets_final = []
        for ii in range(len(ps_df)):
            for x in fdatasets:
                if ps_df['deployment'][ii] in x and ps_df[0][ii] in x:
                    fdatasets_final.append(x)

        # build dictionary of science data from the preferred dataset for each deployment
        print('\nAppending data from files')
        et = []
        sci_vars_dict, y_unit, y_name = cd.append_evaluated_science_data(sDir,
            ps_df, n_streams, r, fdatasets_final, sci_vars_dict0, et, start_time, end_time)

        # get end times of deployments
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        """
        create a data-ranges table and figure for full data time range
        """
        # create a folder to save data ranges
        save_dir_stat = os.path.join(mDir, array, subsite)
        cf.create_dir(save_dir_stat)

        save_fdir = os.path.join(sDir, array, subsite, r, 'data_range')
        cf.create_dir(save_fdir)
        stat_df = pd.DataFrame()


        for m, n in sci_vars_dict.items():
            for sv, vinfo in n['vars'].items():
                print(vinfo['var_name'])
                if len(vinfo['t']) < 1:
                    print('no variable data to plot')
                    continue
                else:
                    sv_units = vinfo['units'][0]
                    fv = vinfo['fv'][0]
                    t = vinfo['t']
                    z = vinfo['values']
                    y = vinfo['pressure']

                # Check if the array is all NaNs
                if sum(np.isnan(z)) == len(z):
                    print('Array of all NaNs - skipping plot.')
                    continue
                # Check if the array is all fill values
                elif len(z[z != fv]) == 0:
                    print('Array of all fill values - skipping plot.')
                    continue
                else:
                    """
                    clean up data
                    """
                    # reject erroneous data
                    dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max = \
                        cf.reject_erroneous_data(r, sv, t, y, z, fv)

                    # reject timestamps from stat analysis
                    Dpath = '{}/{}/{}/{}/{}'.format(sDir, array, subsite, r, 'time_to_exclude')

                    onlyfiles = []
                    for item in os.listdir(Dpath):
                        if not item.startswith('.') and os.path.isfile(os.path.join(Dpath, item)):
                            onlyfiles.append(join(Dpath, item))

                    dre = pd.DataFrame()
                    for nn in onlyfiles:
                        dr = pd.read_csv(nn)
                        dre = dre.append(dr, ignore_index=True)

                    drn = dre.loc[dre['Unnamed: 0'] == vinfo['var_name']]
                    list_time = []
                    for itime in drn.time_to_exclude:
                        ntime = itime.split(', ')
                        list_time.extend(ntime)

                    u_time_list = np.unique(list_time)
                    if len(u_time_list) != 0:
                        t_nospct, z_nospct, y_nospct = cf.reject_suspect_data(dtime, zpressure, ndata, u_time_list)

                    print('{} using {} percentile of data grouped in {} dbar segments'.format(
                        len(zpressure) - len(z_nospct), inpercentile, zcell_size))


                    # reject time range from data portal file export
                    t_portal, z_portal, y_portal = cf.reject_timestamps_dataportal(subsite, r,
                                                                                   t_nospct, y_nospct, z_nospct)

                    print('{} using visual inspection of data'.format(len(z_nospct) - len(z_portal),
                                                                      inpercentile, zcell_size))

                    # reject data in a depth range
                    if zdbar is not None:
                        y_ind = y_portal < zdbar
                        t_array = t_portal[y_ind]
                        y_array = y_portal[y_ind]
                        z_array = z_portal[y_ind]
                    else:
                        y_ind = []
                        t_array = t_portal
                        y_array = y_portal
                        z_array = z_portal
                    print('{} in water depth > {} dbar'.format(len(y_ind), zdbar))


                    if len(y_array) > 0:
                        if m == 'common_stream_placeholder':
                            sname = '-'.join((vinfo['var_name'], r))
                        else:
                            sname = '-'.join((vinfo['var_name'], r, m))

                        """
                        create data ranges for non - pressure data only
                        """

                        if 'pressure' in vinfo['var_name']:
                            pass
                        else:
                            columns = ['tsec', 'dbar', str(vinfo['var_name'])]
                            # create depth ranges
                            min_r = int(round(min(y_array) - zcell_size))
                            max_r = int(round(max(y_array) + zcell_size))
                            ranges = list(range(min_r, max_r, zcell_size))

                            # group data by depth
                            groups, d_groups = gt.group_by_depth_range(t_array, y_array, z_array, columns, ranges)

                            print('writing data ranges for {}'.format(vinfo['var_name']))
                            stat_data = groups.describe()[vinfo['var_name']]
                            stat_data.insert(loc=0, column='parameter', value=sv, allow_duplicates=False)
                            t_deploy = deployments[0]
                            for i in range(len(deployments))[1:len(deployments)]:
                                t_deploy = '{}, {}'.format(t_deploy, deployments[i])
                            stat_data.insert(loc=1, column='deployments', value=t_deploy, allow_duplicates=False)

                        stat_df = stat_df.append(stat_data, ignore_index=True)

                        """
                        plot full time range free from errors and suspect data
                        """

                        clabel = sv + " (" + sv_units + ")"
                        ylabel = (y_name[0][0] + " (" + y_unit[0][0] + ")")
                        title = ' '.join((r, m))


                        # plot non-erroneous -suspect data
                        fig, ax, bar = pf.plot_xsection(subsite, t_array, y_array, z_array,
                                                                    clabel, ylabel, inpercentile=None, stdev=None)

                        ax.set_title(title, fontsize=9)
                        leg_text = (
                            'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}]'.format(
                                len(z) - lenfv, len(z) - lennan, len(z) - lenev, lengr, global_min, global_max) + '\n' +
                            ('removed {} in the upper and lower {} percentile of data grouped in {} dbar segments'.format(
                                                            len(zpressure) - len(z_nospct), inpercentile, zcell_size)),)

                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)


                        for ii in range(len(end_times)):
                            ax.axvline(x=end_times[ii], color='b', linestyle='--', linewidth=.8)
                            ax.text(end_times[ii], min(y_array)-5, 'End' + str(deployments[ii]),
                                                   fontsize=6, style='italic',
                                                   bbox=dict(boxstyle='round',
                                                             ec=(0., 0.5, 0.5),
                                                             fc=(1., 1., 1.),
                                                             ))

                        fig.tight_layout()
                        sfile = '_'.join(('data_range', sname))
                        pf.save_fig(save_fdir, sfile)

            # write stat file
            stat_df.to_csv('{}/{}_data_ranges.csv'.format(save_dir_stat, r), index=True, float_format='%11.6f')


if __name__ == '__main__':
    '''
        define time range: 
        set to None if plotting all data
        set to dt.datetime(yyyy, m, d, h, m, s) for specific dates
        '''
    start_time = dt.datetime(2016, 1, 1) #10/16/2015	05/07/2016
    end_time = dt.datetime(2016, 1, 15)
    # start_time = None
    # end_time = None

    '''
    define filters standard deviation, percentile, depth range
    '''
    n_std = None
    inpercentile = 5
    zdbar = None

    '''
    define the depth cell_size for data grouping 
    '''
    zcell_size = 10

    '''
        define plot type, save-directory name and URL where data files live 
    '''
    mainP = '/Users/leila/Documents/NSFEduSupport/'
    mDir = mainP + 'github/data-review-tools/data_review/data_ranges'
    sDir = mainP + 'review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021350-CE09OSPM-WFP01-04-FLORTK000-telemetered-flort_sample/catalog.html']

    main(url_list, sDir, mDir, zcell_size, zdbar, start_time, end_time)
