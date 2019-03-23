#!/usr/bin/env python
"""
Created on Jan 6 2019

@author: Leila Belabbassi
@brief: This script is used to compare year-to-year timeseries plots for a science variable and provide statistical data description:
Figure 1 - shows data and the mean and std lines calculated using the rolling window method,
Figure 2 - shows data histograms and a table with basic statistics
"""

import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt
import os
import pandas as pd
import itertools
import numpy as np
import xarray as xr
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import pyplot
from matplotlib import colors as mcolors
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
from matplotlib.ticker import MaxNLocator
from statsmodels.nonparametric.kde import KDEUnivariate


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
color_names = [name for hsv, name in colors.items()]

def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None

def save_dir_path(ms_list):
    ms_list_check = [s.split('-')[0] for s in ms_list]
    n_ms = [ms_list_check.count(s) for s in np.unique(ms_list_check)]
    ms_dict = dict({'ms_unique': ms_list_check, 'ms_count': n_ms})
    return ms_dict

def main(sDir, url_list):
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

        ps_df, n_streams = cf.get_preferred_stream_info(r)

        # get end times of deployments
        dr_data = cf.refdes_datareview_json(r)
        deployments = []
        end_times = []
        for index, row in ps_df.iterrows():
            deploy = row['deployment']
            deploy_info = get_deployment_information(dr_data, int(deploy[-4:]))
            deployments.append(int(deploy[-4:]))
            end_times.append(pd.to_datetime(deploy_info['stop_date']))

        # filter datasets
        datasets = []
        for u in url_list:
            print(u)
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        main_sensor = r.split('-')[-1]
        fdatasets = cf.filter_collocated_instruments(main_sensor, datasets)
        fdatasets = cf.filter_other_streams(r, ms_list, fdatasets)

        methodstream = []
        for f in fdatasets:
            methodstream.append('-'.join((f.split('/')[-2].split('-')[-2], f.split('/')[-2].split('-')[-1])))

        ms_dict = save_dir_path(ms_list)
        for ms in np.unique(methodstream):
            fdatasets_sel = [x for x in fdatasets if ms in x]
            check_ms = ms.split('-')[1]
            if 'recovered' in check_ms:
                check_ms = check_ms.split('_recovered')[0]

            if ms_dict['ms_count'][ms_dict['ms_unique'] == ms.split('-')[0]] == 1:
                save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_yearly_plot',
                                        ms.split('-')[0])
            else:
                save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_yearly_plot',
                                    ms.split('-')[0], check_ms)
            cf.create_dir(save_dir)

            stream_sci_vars_dict = dict()
            for x in dr_data['instrument']['data_streams']:
                dr_ms = '-'.join((x['method'], x['stream_name']))
                if ms == dr_ms:
                    stream_sci_vars_dict[dr_ms] = dict(vars=dict())
                    sci_vars = dict()
                    for y in x['stream']['parameters']:
                        if y['data_product_type'] == 'Science Data':
                            sci_vars.update({y['name']: dict(db_units=y['unit'])})
                    if len(sci_vars) > 0:
                        stream_sci_vars_dict[dr_ms]['vars'] = sci_vars

            sci_vars_dict = cd.initialize_empty_arrays(stream_sci_vars_dict, ms)
            print('\nAppending data from files: {}'.format(ms))
            for fd in fdatasets_sel:
                ds = xr.open_dataset(fd, mask_and_scale=False)
                print(fd)
                for var in list(sci_vars_dict[ms]['vars'].keys()):
                    sh = sci_vars_dict[ms]['vars'][var]
                    try:
                        ds[var]
                        if ds[var].units == sh['db_units']:
                            if ds[var]._FillValue not in sh['fv']:
                                sh['fv'].append(ds[var]._FillValue)
                            if ds[var].units not in sh['units']:
                                sh['units'].append(ds[var].units)
                            tD = ds['time'].values
                            varD = ds[var].values
                            sh['t'] = np.append(sh['t'], tD)
                            sh['values'] = np.append(sh['values'], varD)
                    except KeyError:
                        print('KeyError: ', var)

            print('\nPlotting data')
            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print(sv)
                    if len(vinfo['t']) < 1:
                        print('no variable data to plot')
                    else:
                        sv_units = vinfo['units'][0]
                        fv = vinfo['fv'][0]
                        t0 = pd.to_datetime(min(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        t1 = pd.to_datetime(max(vinfo['t'])).strftime('%Y-%m-%dT%H:%M:%S')
                        x = vinfo['t']
                        y = vinfo['values']

                        # reject NaNs
                        nan_ind = ~np.isnan(y)
                        x_nonan = x[nan_ind]
                        y_nonan = y[nan_ind]

                        # reject fill values
                        fv_ind = y_nonan != vinfo['fv'][0]
                        x_nonan_nofv = x_nonan[fv_ind]
                        y_nonan_nofv = y_nonan[fv_ind]

                        # reject extreme values
                        Ev_ind = cf.reject_extreme_values(y_nonan_nofv)
                        y_nonan_nofv_nE = y_nonan_nofv[Ev_ind]
                        x_nonan_nofv_nE = x_nonan_nofv[Ev_ind]

                        # reject values outside global ranges:
                        global_min, global_max = cf.get_global_ranges(r, sv)
                        print('global ranges: ', global_min, global_max)
                        if global_min and global_max:
                            gr_ind = cf.reject_global_ranges(y_nonan_nofv_nE, global_min, global_max)
                            y_nonan_nofv_nE_nogr = y_nonan_nofv_nE[gr_ind]
                            x_nonan_nofv_nE_nogr = x_nonan_nofv_nE[gr_ind]
                        else:
                            y_nonan_nofv_nE_nogr = y_nonan_nofv_nE
                            x_nonan_nofv_nE_nogr = x_nonan_nofv_nE

                        if len(y_nonan_nofv_nE_nogr) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((r, sv))
                                print(var, 'empty array')
                            else:
                                sname = '-'.join((r, m, sv))

                            # group data by year
                            groups, g_data = gt.group_by_time_range(x_nonan_nofv_nE_nogr, y_nonan_nofv_nE_nogr, 'A')

                            # create bins
                            # groups_min = min(groups.describe()['DO']['min'])
                            # lower_bound = int(round(groups_min))
                            # groups_max = max(groups.describe()['DO']['max'])
                            # if groups_max < 1:
                            #     upper_bound = 1
                            #     step_bound = 1
                            # else:
                            #     upper_bound = int(round(groups_max + (groups_max / 50)))
                            #     step_bound = int(round((groups_max - groups_min) / 10))
                            #
                            # if step_bound == 0:
                            #     step_bound += 1
                            #
                            # if (upper_bound - lower_bound) == step_bound:
                            #     lower_bound -= 1
                            #     upper_bound += 1
                            # if (upper_bound - lower_bound) < step_bound:
                            #     print('<')
                            #     step_bound = int(round(step_bound / 10))
                            # print(lower_bound, upper_bound, step_bound)
                            # bin_range = list(range(lower_bound, upper_bound, step_bound))
                            # print(bin_range)

                            # preparing color palette
                            colors = color_names[:len(groups)]

                            # colors = [color['color'] for color in
                            #           list(pyplot.rcParams['axes.prop_cycle'][:len(groups)])]


                            fig0, ax0 = pyplot.subplots(nrows=2, ncols=1)

                            # subplot for  histogram and basic statistics table
                            ax0[1].axis('off')
                            ax0[1].axis('tight')
                            the_table = ax0[1].table(cellText=groups.describe().round(2).values,
                                                     rowLabels=groups.describe().index.year,
                                                     rowColours=colors,
                                                     colLabels=groups.describe().columns.levels[1],
                                                     loc='center')
                            the_table.set_fontsize(5)

                            # subplot for data
                            fig, ax = pyplot.subplots(nrows=len(groups), ncols=1, sharey=True)
                            if len(groups) == 1:
                                ax=[ax]
                            t = 1
                            for ny in range(len(groups)):

                                # prepare data for plotting
                                y_data = g_data[ny + (t + 1)].dropna(axis=0)
                                x_time = g_data[ny+t].dropna(axis=0)
                                t += 1

                                n_year = x_time[0].year

                                col_name = str(n_year)

                                serie_n = pd.DataFrame(columns=[col_name], index=x_time)
                                serie_n[col_name] = list(y_data[:])

                                # plot histogram
                                # serie_n.plot.hist(ax=ax0[0], bins=bin_range,
                                #                   histtype='bar', color=colors[ny], stacked=True)
                                serie_n.plot.kde(ax=ax0[0], color=colors[ny])
                                ax0[0].legend(fontsize=8, bbox_to_anchor=(0., 1.12, 1., .102), loc=3,
                                              ncol=len(groups), mode="expand", borderaxespad=0.)

                                # ax0[0].set_xticks(bin_range)
                                ax0[0].set_xlabel('Observation Ranges', fontsize=8)
                                ax0[0].set_ylabel('Density', fontsize=8) #'Number of Observations'
                                ax0[0].set_title(ms.split('-')[0] + ' (' + sv + ', ' + sv_units+')' +
                                                 '  Kernel Density Estimates', fontsize=8)

                                # plot data
                                serie_n.plot(ax=ax[ny], linestyle='None', marker='.', markersize=0.5, color=colors[ny])
                                ax[ny].legend().set_visible(False)

                                # plot Mean and Standard deviation
                                ma = serie_n.rolling('86400s').mean()
                                mstd = serie_n.rolling('86400s').std()

                                ax[ny].plot(ma.index, ma[col_name].values, 'k', linewidth=0.15)
                                ax[ny].fill_between(mstd.index, ma[col_name].values - 2 * mstd[col_name].values,
                                                    ma[col_name].values + 2 * mstd[col_name].values,
                                                    color='b', alpha=0.2)

                                # prepare the time axis parameters
                                datemin = datetime.date(n_year, 1, 1)
                                datemax = datetime.date(n_year, 12, 31)
                                ax[ny].set_xlim(datemin, datemax)
                                xlocator = mdates.MonthLocator()  # every month
                                myFmt = mdates.DateFormatter('%m')
                                ax[ny].xaxis.set_minor_locator(xlocator)
                                ax[ny].xaxis.set_major_formatter(myFmt)

                                # prepare the time axis parameters
                                # ax[ny].set_yticks(bin_range)
                                ylocator = MaxNLocator(prune='both', nbins=3)
                                ax[ny].yaxis.set_major_locator(ylocator)

                                # format figure
                                ax[ny].tick_params(axis='both', color='r', labelsize=7, labelcolor='m')

                                if ny < len(groups)-1:
                                    ax[ny].tick_params(which='both', pad=0.1, length=1, labelbottom=False)
                                    ax[ny].set_xlabel(' ')
                                else:
                                    ax[ny].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                                       pad=0.1, length=1, rotation=0)
                                    ax[ny].set_xlabel('Months', rotation=0, fontsize=8, color='b')

                                ax[ny].set_ylabel(n_year, rotation=0, fontsize=8, color='b', labelpad=20)
                                ax[ny].yaxis.set_label_position("right")

                                if ny == 0:
                                    if global_min and global_max:

                                        ax[ny].set_title(sv + '( '+ sv_units + ') -- Global Range: [' + str(int(global_min)) +
                                                         ',' + str(int(global_max)) + '] \n'
                                                         'Plotted: Data, Mean and 2STD (Method: One day rolling window calculations) \n',
                                                         fontsize=8)
                                    else:
                                        ax[ny].set_title(
                                         sv + '( ' + sv_units + ') -- Global Range: [] \n'
                                        'Plotted: Data, Mean and 2STD (Method: One day rolling window calculations) \n',
                                            fontsize=8)

                                # plot global ranges
                                # ax[ny].axhline(y=global_min, color='r', linestyle='--', linewidth=.6)
                                # ax[ny].axhline(y=global_max, color='r', linestyle='--', linewidth=.6)

                                # mark deployment end times on figure
                                ymin, ymax = ax[ny].get_ylim()
                                dep = 1
                                for etimes in end_times:
                                    if etimes.year == n_year:
                                        ax[ny].axvline(x=etimes, color='b', linestyle='--', linewidth=.6)
                                        ax[ny].text(etimes, ymin, 'End' + str(dep), fontsize=6, style='italic',
                                                    bbox=dict(boxstyle='round',
                                                              ec=(0., 0.5, 0.5),
                                                              fc=(1., 1., 1.))
                                                    )
                                    dep += 1

                                # ax[ny].set_ylim(5, 12)

                            # save figure to a file
                            sfile = '_'.join(('all', sname))
                            save_file = os.path.join(save_dir, sfile)
                            fig.savefig(str(save_file), dpi=150)

                            sfile = '_'.join(('Statistics', sname))
                            save_file = os.path.join(save_dir, sfile)
                            fig0.savefig(str(save_file), dpi=150)

                            pyplot.close()

if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = []


    main(sDir, url_list)