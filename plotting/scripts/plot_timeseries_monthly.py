#!/usr/bin/env python
"""
Created on Jan 6 2019

@author: Leila Belabbassi
@brief: This script is used to compare monthly timeseries plots for a science variable and
provide statistical data description:
Figure 1 - shows data and the mean and std lines calculated using the rolling window method,
Figure 2 - shows data histograms and a table with basic statistics
"""

import os
import pandas as pd
import itertools
import numpy as np
import xarray as xr
import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt

from matplotlib import pyplot
from matplotlib import colors as mcolors

import matplotlib.gridspec as gridspec
import datetime
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import calendar
from calendar import monthrange
from matplotlib.ticker import MaxNLocator, LinearLocator


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
color_names = [name for hsv, name in colors.items()]


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None


def main(sDir, url_list):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        if rd not in rd_list:
            rd_list.append(rd)

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
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)
        datasets = list(itertools.chain(*datasets))
        main_sensor = r.split('-')[-1]
        fdatasets = cf.filter_collocated_instruments(main_sensor, datasets)
        methodstream = []
        for f in fdatasets:
            methodstream.append('-'.join((f.split('/')[-2].split('-')[-2], f.split('/')[-2].split('-')[-1])))

        for ms in np.unique(methodstream):
            fdatasets_sel = [x for x in fdatasets if ms in x]

            check_ms = ms.split('-')[1]
            if 'recovered' in check_ms:
                check_ms = check_ms.split('_recovered')

            save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_monthly_plot',
                                    check_ms[0], ms.split('-')[0])
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
                for var in list(sci_vars_dict[ms]['vars'].keys()):
                    sh = sci_vars_dict[ms]['vars'][var]
                    if ds[var].units == sh['db_units']:
                        if ds[var]._FillValue not in sh['fv']:
                            sh['fv'].append(ds[var]._FillValue)
                        if ds[var].units not in sh['units']:
                            sh['units'].append(ds[var].units)
                        tD = ds['time'].values
                        varD = ds[var].values
                        sh['t'] = np.append(sh['t'], tD)
                        sh['values'] = np.append(sh['values'], varD)

            print('\nPlotting data')
            for m, n in sci_vars_dict.items():
                for sv, vinfo in n['vars'].items():
                    print(sv)
                    if len(vinfo['t']) < 1:
                        print('no variable data to plot')
                    else:
                        sv_units = vinfo['units'][0]
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
                        gr_ind = cf.reject_global_ranges(y_nonan_nofv_nE, global_min, global_max)
                        y_nonan_nofv_nE_nogr = y_nonan_nofv_nE[gr_ind]
                        x_nonan_nofv_nE_nogr = x_nonan_nofv_nE[gr_ind]

                        title = ' '.join((r, ms.split('-')[0]))

                        if len(y_nonan_nofv) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((r, sv))
                            else:
                                sname = '-'.join((r, m, sv))

                            # 1st group by year
                            ygroups, gy_data = gt.group_by_timerange(x_nonan_nofv_nE_nogr, y_nonan_nofv_nE_nogr, 'A')

                            tn = 1
                            for n in range(len(ygroups)):
                                x_time = gy_data[n+tn].dropna(axis=0)
                                y_data = gy_data[n+(tn+1)].dropna(axis=0)
                                y_data = y_data.astype(float)
                                # 2nd group by month
                                mgroups, gm_data = gt.group_by_timerange(x_time.values, y_data.values, 'M')

                                x_year = x_time[0].year
                                print(x_year)
                                #
                                # create bins for histogram
                                mgroups_min = min(mgroups.describe()['DO']['min'])
                                mgroups_max = max(mgroups.describe()['DO']['max'])
                                lower_bound = int(round(mgroups_min))
                                upper_bound = int(round(mgroups_max + (mgroups_max / 50)))
                                step_bound = int(round((mgroups_max - mgroups_min) / 10))

                                lower_bound = int(round(global_min))
                                upper_bound = int(round(global_max + (global_max / 50)))
                                step_bound = int(round((global_max - global_min) / 10))

                                if step_bound == 0:
                                    step_bound += 1

                                if (upper_bound - lower_bound) == step_bound:
                                    lower_bound -= 1
                                    upper_bound += 1
                                if (upper_bound - lower_bound) < step_bound:
                                    step_bound = int(round(step_bound / 10))

                                bin_range = list(range(lower_bound, upper_bound, step_bound))
                                print(bin_range)

                                # create color palette

                                colors = color_names[:len(mgroups)]
                                print('1--- ', len(colors))
                                print(colors)


                                fig0, ax0 = pyplot.subplots(nrows=2, ncols=1)

                                # # subplot for  histogram and basic statistics table
                                ax0[0].axis('off')
                                ax0[0].axis('tight')

                                the_table = ax0[0].table(cellText=mgroups.describe().round(2).values,
                                                         rowLabels=mgroups.describe().index.month,
                                                         rowColours=colors,
                                                         colLabels=mgroups.describe().columns.levels[1], loc='center')
                                the_table.set_fontsize(5)

                                fig, ax = pyplot.subplots(nrows=12, ncols=1, sharey=True)

                                for kk in list(range(0, 12)):
                                    ax[kk].tick_params(axis='both', which='both', color='r', labelsize=7,
                                                       labelcolor='m', rotation=0, pad=0.1, length=1)
                                    month_name = calendar.month_abbr[kk + 1]
                                    ax[kk].set_ylabel(month_name, rotation=0, fontsize=8, color='b', labelpad=20)
                                    if kk == 0:
                                        ax[kk].set_title(str(x_year) + '\n ' + sv + " (" + sv_units + ")" +
                                                         ' Global Range: [' + str(int(global_min)) + ',' + str(int(global_max)) + ']' +
                                                         '\n End of deployments are marked with a vertical line \n ' +
                                                         'Plotted: Data, Mean and STD (Method: 1 day' +
                                                         ' rolling window calculations)',
                                                         fontsize=8)

                                    if kk < 11:
                                        ax[kk].tick_params(labelbottom=False)
                                    if kk == 11:
                                        ax[kk].set_xlabel('Days', rotation=0, fontsize=8, color='b')

                                tm = 1
                                for mt in range(len(mgroups)):
                                    x_time = gm_data[mt+tm].dropna(axis=0)
                                    y_data = gm_data[mt+(tm+1)].dropna(axis=0)

                                    if len(x_time) == 0:
                                        # ax[plt_index].tick_params(which='both', labelbottom=False, labelleft=False,
                                        #                    pad=0.1, length=1)
                                        continue

                                    x_month = x_time[0].month
                                    col_name = str(x_month)

                                    series_m = pd.DataFrame(columns=[col_name], index=x_time)
                                    series_m[col_name] = list(y_data[:])


                                    # serie_n.plot.hist(ax=ax0[0], bins=bin_range,
                                    #                   histtype='bar', color=colors[ny], stacked=True)
                                    series_m.plot.kde(ax=ax0[0], color=colors[mt])
                                    ax0[0].legend(fontsize=8, bbox_to_anchor=(0., 1.12, 1., .102), loc=3,
                                                  ncol=len(mgroups), mode="expand", borderaxespad=0.)

                                    # ax0[0].set_xticks(bin_range)
                                    ax0[0].set_xlabel('Observation Ranges' + ' (' + sv + ', ' + sv_units + ')', fontsize=8)
                                    ax0[0].set_ylabel('Density', fontsize=8)  # 'Number of Observations'
                                    ax0[0].set_title('Kernel Density Estimates', fontsize=8)
                                    ax0[0].tick_params(which='both', labelsize=7, pad=0.1, length=1, rotation=0)

                                    plt_index = x_month - 1

                                    # Plot data
                                    series_m.plot(ax=ax[plt_index], linestyle='None', marker='.', markersize=1)
                                    ax[plt_index].legend().set_visible(False)

                                    ma = series_m.rolling('86400s').mean()
                                    mstd = series_m.rolling('86400s').std()

                                    ax[plt_index].plot(ma.index, ma[col_name].values, 'b')
                                    ax[plt_index].fill_between(mstd.index, ma[col_name].values-3*mstd[col_name].values,
                                                               ma[col_name].values+3*mstd[col_name].values,
                                                               color='b', alpha=0.2)

                                    # prepare the time axis parameters
                                    mm, nod = monthrange(x_year, x_month)
                                    datemin = datetime.date(x_year, x_month, 1)
                                    datemax = datetime.date(x_year, x_month, nod)
                                    ax[plt_index].set_xlim(datemin, datemax)
                                    xlocator = mdates.DayLocator()  # every day
                                    myFmt = mdates.DateFormatter('%d')
                                    ax[plt_index].xaxis.set_major_locator(xlocator)
                                    ax[plt_index].xaxis.set_major_formatter(myFmt)
                                    ax[plt_index].xaxis.set_minor_locator(pyplot.NullLocator())
                                    ax[plt_index].xaxis.set_minor_formatter(pyplot.NullFormatter())

                                    # data_min = min(ma.DO_n.dropna(axis=0) - 5 * mstd.DO_n.dropna(axis=0))
                                    # 0data_max = max(ma.DO_n.dropna(axis=0) + 5 * mstd.DO_n.dropna(axis=0))
                                    # ax[plt_index].set_ylim([data_min, data_max])

                                    ylocator = MaxNLocator(prune='both', nbins=3)
                                    ax[plt_index].yaxis.set_major_locator(ylocator)


                                    if x_month != 12:
                                        ax[plt_index].tick_params(which='both', labelbottom=False, pad=0.1, length=1)
                                        ax[plt_index].set_xlabel(' ')
                                    else:
                                        ax[plt_index].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                                           pad=0.1, length=1, rotation=0)
                                        ax[plt_index].set_xlabel('Days', rotation=0, fontsize=8, color='b')

                                    dep = 1
                                    for etimes in end_times:
                                        ax[plt_index].axvline(x=etimes, color='b', linestyle='--', linewidth=.8)
                                        if ma[col_name].values.any():
                                            ax[plt_index].text(etimes, max(ma[col_name].dropna(axis=0)), 'End' + str(dep),
                                                        fontsize=6, style='italic',
                                                        bbox=dict(boxstyle='round',
                                                                  ec=(0., 0.5, 0.5),
                                                                  fc=(1., 1., 1.),
                                                                  ))
                                        else:
                                            ax[plt_index].text(etimes, min(series_m['DO_n']), 'End' + str(dep),
                                                        fontsize=6, style='italic',
                                                        bbox=dict(boxstyle='round',
                                                                  ec=(0., 0.5, 0.5),
                                                                  fc=(1., 1., 1.),
                                                                  ))
                                        dep += 1
                                    tm += 1
                                tn += 1


                                # pyplot.show()
                                sfile = '_'.join((str(x_year), sname))
                                save_file = os.path.join(save_dir, sfile)
                                fig.savefig(str(save_file), dpi=150)

                                sfile = '_'.join(('Statistics', str(x_year), sname))
                                save_file = os.path.join(save_dir, sfile)
                                fig0.savefig(str(save_file), dpi=150)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000821-CP03ISSM-SBD11-06-METBKA000-telemetered-metbk_a_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T000115-CP03ISSM-SBD11-06-METBKA000-recovered_host-metbk_a_dcl_instrument_recovered/catalog.html']

    main(sDir, url_list)
