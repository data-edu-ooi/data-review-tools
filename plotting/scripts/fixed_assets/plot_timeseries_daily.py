#!/usr/bin/env python
"""
Created on Dec 14 2018

@author: Lori Garzio
@brief: This script is used create two timeseries plots of raw and science variables for all deployments of a reference
designator by delivery method: 1) plot all data, 2) plot data, omitting outliers beyond 5 standard deviations.
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
import datetime
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import calendar
from calendar import monthrange

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
            save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_daily_plots', ms.split('-')[0])
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


                        if len(y_nonan_nofv) > 0:
                            if m == 'common_stream_placeholder':
                                sname = '-'.join((r, sv))
                            else:
                                sname = '-'.join((r, m, sv))

                            # 1st group by year
                            ygroups, gy_data = gt.group_by_timerange(x_nonan_nofv_nE_nogr, y_nonan_nofv_nE_nogr, 'A')

                            tn = 1
                            for n in range(len(ygroups)):
                                x_time = gy_data[n + tn].dropna(axis=0)
                                y_data = gy_data[n + (tn + 1)].dropna(axis=0)

                                # 2nd group by month
                                mgroups, gm_data = gt.group_by_timerange(x_time.values, y_data.values, 'M')

                                if len(x_time) == 0:
                                    continue

                                td = 1
                                for jj in range(len(mgroups)):
                                    x_time = gm_data[jj + td].dropna(axis=0)
                                    y_data = gm_data[jj + (td + 1)].dropna(axis=0)

                                    if len(x_time) == 0:
                                        continue

                                    # 3rd group by day
                                    dgroups, gd_data = gt.group_by_timerange(x_time.values, y_data.values, 'D')

                                    x_year = x_time[0].year
                                    x_month = x_time[0].month
                                    month_name = calendar.month_abbr[x_month]
                                    print(x_year, x_month)

                                    sfile = '_'.join((str(x_year), str(x_month), sname))

                                    # prepare plot layout

                                    fig, ax = pyplot.subplots(nrows=7, ncols=5, sharey=True)
                                    title_in = month_name + '-' + str(x_year) + \
                                                  ' calendar days \n Parameter: ' + \
                                                  sv + " (" + sv_units + ")"

                                    ax[0][2].text(0.5, 1.5, title_in,
                                                  horizontalalignment='center',
                                                  fontsize=8,
                                                  transform=ax[0][2].transAxes)
                                    num_i = 0
                                    day_i = {}
                                    for kk in list(range(0, 7)):
                                        for ff in list(range(0, 5)):
                                            num_i += 1
                                            day_i[num_i] = [kk, ff]
                                            ax[kk][ff].tick_params(axis='both', which='both', color='r', labelsize=7,
                                                                   labelcolor='m', rotation=0)

                                            ax[kk][ff].text(0.1, 0.75, str(num_i),
                                                            horizontalalignment='center',
                                                            fontsize=7,
                                                            transform=ax[kk][ff].transAxes,
                                                            bbox=dict(boxstyle="round",
                                                                      ec=(0., 0.5, 0.5),
                                                                      fc=(1., 1., 1.),
                                                                      ))

                                            if kk is not 6:
                                                ax[kk][ff].tick_params(labelbottom=False)
                                            if ff is not 0:
                                                ax[kk][ff].tick_params(labelright=False)

                                            if kk is 6 and ff is 0:
                                                ax[kk][ff].set_xlabel('Hours', rotation=0, fontsize=8, color='b')

                                            if kk is 6 and ff in list(range(1, 5)):
                                                fig.delaxes(ax[kk][ff])


                                    tm = 1
                                    for mt in range(len(dgroups)):
                                        x_time = gd_data[mt + tm].dropna(axis=0)
                                        y_DO = gd_data[mt + (tm + 1)].dropna(axis=0)

                                        series_m = pd.DataFrame(columns=['DO_n'], index=x_time)
                                        series_m['DO_n'] = list(y_DO[:])

                                        if len(x_time) == 0:
                                            continue

                                        x_day = x_time[0].day

                                        print(x_time[0].year, x_time[0].month, x_day)

                                        i0 = day_i[x_day][0]
                                        i1 = day_i[x_day][1]

                                        # Plot data
                                        series_m.plot(ax=ax[i0][i1], linestyle='None', marker='.', markersize=1)
                                        ax[i0][i1].legend().set_visible(False)

                                        ma = series_m.rolling('3600s').mean()
                                        mstd = series_m.rolling('3600s').std()

                                        ax[i0][i1].plot(ma.index, ma.DO_n, 'b', linewidth=0.25)
                                        ax[i0][i1].fill_between(mstd.index, ma.DO_n-3*mstd.DO_n, ma.DO_n+3*mstd.DO_n,
                                                                color='b', alpha=0.2)

                                        # prepare the time axis parameters
                                        datemin = datetime.datetime(x_year, x_month, x_day, 0)
                                        datemax = datetime.datetime(x_year, x_month, x_day, 23)

                                        ax[i0][i1].set_xlim(datemin, datemax)
                                        xLocator = mdates.HourLocator(interval=4)  # every hour
                                        myFmt = mdates.DateFormatter('%H')
                                        ax[i0][i1].xaxis.set_minor_locator(xLocator)
                                        ax[i0][i1].xaxis.set_minor_formatter(myFmt)
                                        ax[i0][i1].xaxis.set_major_locator(pyplot.NullLocator())
                                        ax[i0][i1].xaxis.set_major_formatter(pyplot.NullFormatter())
                                        yLocator = MaxNLocator(prune='both', nbins=3)
                                        ax[i0][i1].yaxis.set_major_locator(yLocator)

                                        if x_day is not 31:
                                            ax[i0][i1].tick_params(labelbottom=False)
                                            ax[i0][i1].set_xlabel(' ')
                                        else:
                                            ax[i0][i1].tick_params(which='both', color='r', labelsize=7,
                                                                   labelcolor='m', length=0.1, pad=0.1)
                                            ax[i0][i1].set_xlabel('Hours', rotation=0, fontsize=8, color='b')



                                        ymin, ymax = ax[i0][i1].get_ylim()
                                        dep = 1
                                        for etimes in end_times:
                                            ax[i0][i1].axvline(x=etimes, color='b', linestyle='--', linewidth=.6)
                                            ax[i0][i1].text(etimes, ymin+50, str(dep),
                                                            fontsize=6, style='italic',
                                                            bbox=dict(boxstyle="round",
                                                                      ec=(0., 0.5, 0.5),
                                                                      fc=(1., 1., 1.),
                                                                      ))

                                            dep += 1
                                        tm += 1
                                    td += 1
                                    pf.save_fig(save_dir, sfile)
                                tn += 1


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154700-CE06ISSM-RID16-03-CTDBPC000-recovered_host-ctdbp_cdef_dcl_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154713-CE06ISSM-RID16-03-CTDBPC000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154849-CE06ISSM-RID16-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument/catalog.html']

        # [
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163408-CE06ISSM-RID16-03-DOSTAD000-recovered_host-dosta_abcdjm_ctdbp_dcl_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163419-CE06ISSM-RID16-03-DOSTAD000-recovered_inst-dosta_abcdjm_ctdbp_instrument_recovered/catalog.html',
        #         'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163558-CE06ISSM-RID16-03-DOSTAD000-telemetered-dosta_abcdjm_ctdbp_dcl_instrument/catalog.html'
        #         ]
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163751-CE09OSPM-WFP01-02-DOFSTK000-recovered_wfp-dofst_k_wfp_instrument_recovered/catalog.html',
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163824-CE09OSPM-WFP01-02-DOFSTK000-telemetered-dofst_k_wfp_instrument/catalog.html',
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163845-CE09OSSM-RID27-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html',
    # 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163907-CE09OSSM-RID27-04-DOSTAD000-telemetered-dosta_abcdjm_dcl_instrument/catalog.html'
    #
    main(sDir, url_list)
