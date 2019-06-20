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


def main(url_list, sDir, mDir, zcell_size, zdbar, start_time, end_time, inpercentile):

    """""
    URL : path to instrument data by methods
    sDir : path to the directory on your machine to save plots
    mDir : path to the directory on your machine to save data ranges
    zcell_size : depth cell size to group data
    zdbar : define depth where suspect data are identified
    start_time : select start date to slice timeseries
    end_time : select end date to slice timeseries
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
        sci_vars_dict, y_unit, y_name, l0 = cd.append_evaluated_science_data(
                                sDir, ps_df, n_streams, r, fdatasets_final, sci_vars_dict0, zdbar, start_time, end_time)

        # get end times of deployments
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = cf.get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        # create data range output folders
        save_dir_stat = os.path.join(mDir, array, subsite)
        cf.create_dir(save_dir_stat)
        # create plots output folder
        save_fdir = os.path.join(sDir, array, subsite, r, 'data_range')
        cf.create_dir(save_fdir)
        stat_df = pd.DataFrame()

        """
        create data ranges csv file and figures
        """
        for m, n in sci_vars_dict.items():
            for sv, vinfo in n['vars'].items():
                print('\n' + vinfo['var_name'])
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

                    if len(y) > 0:
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
                            min_r = int(round(min(y) - zcell_size))
                            max_r = int(round(max(y) + zcell_size))
                            ranges = list(range(min_r, max_r, zcell_size))

                            # group data by depth
                            groups, d_groups = gt.group_by_depth_range(t, y, z, columns, ranges)

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

                        t_eng = None
                        m_water_depth = None

                        # plot non-erroneous -suspect data
                        fig, ax, bar = pf.plot_xsection(subsite, t, y, z, clabel, ylabel, t_eng, m_water_depth,
                                                        inpercentile, stdev=None)

                        title0 = 'Data colored using the upper and lower {}th percentile.'.format(inpercentile)
                        ax.set_title(r+'\n'+title0, fontsize=9)
                        leg_text = ('{} % erroneous values removed after Human In the Loop review'.format(
                                                                                                    (len(t)/l0) * 100),)
                        ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)


                        for ii in range(len(end_times)):
                            ax.axvline(x=end_times[ii], color='b', linestyle='--', linewidth=.8)
                            ax.text(end_times[ii], min(y)-5, 'End' + str(deployments[ii]),
                                                   fontsize=6, style='italic',
                                                   bbox=dict(boxstyle='round',
                                                             ec=(0., 0.5, 0.5),
                                                             fc=(1., 1., 1.),
                                                             ))

                        # fig.tight_layout()
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
    # start_time = dt.datetime(2014, 10, 16) #10/16/2015	05/07/2016
    # end_time = dt.datetime(2016, 5, 7)
    start_time = None
    end_time = None

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
    #glider = 'no' # options: yes, no
    mainP = '/Users/leila/Documents/NSFEduSupport/'
    mDir = mainP + 'github/data-review-tools/data_review/data_ranges'
    sDir = mainP + 'review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021222-CE09OSPM-WFP01-04-FLORTK000-recovered_wfp-flort_sample/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021350-CE09OSPM-WFP01-04-FLORTK000-telemetered-flort_sample/catalog.html']

    main(url_list, sDir, mDir, zcell_size, zdbar, start_time, end_time, inpercentile)
