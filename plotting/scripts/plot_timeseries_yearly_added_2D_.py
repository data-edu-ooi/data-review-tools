#!/usr/bin/env python
"""
Created on Jan 6 2019

@author: Leila Belabbassi
@brief: This script is used to compare year-to-year timeseries plots for a non-science variable:
Figure 1 - data plot,
"""

import functions.common as cf
import functions.plotting as pf
import functions.combine_datasets as cd
import functions.group_by_timerange as gt
import functions.split_by_timegap as fsplt
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
from pandas.plotting import scatter_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
color_names = [name for hsv, name in colors.items()]


def in_list(x, ix):
    y = [el for el in x if any(ignore in el for ignore in ix)]
    return y

def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None

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

        for ms in np.unique(methodstream):
            fdatasets_sel = [x for x in fdatasets if ms in x]
            save_dir = os.path.join(sDir, array, subsite, r, 'timeseries_yearly_plot', ms.split('-')[0])
            cf.create_dir(save_dir)

            stream_sci_vars_dict = dict()
            for x in dr_data['instrument']['data_streams']:
                dr_ms = '-'.join((x['method'], x['stream_name']))
                if ms == dr_ms:
                    stream_sci_vars_dict[dr_ms] = dict(vars=dict())
                    sci_vars = dict()
                    for y in x['stream']['parameters']:
                        if 'light' in y['name']:
                            sci_vars.update({y['name']: dict(db_units=y['unit'])})
                        # if y['data_product_type'] == 'Science Data':
                        #     sci_vars.update({y['name']: dict(db_units=y['unit'])})
                    if len(sci_vars) > 0:
                        stream_sci_vars_dict[dr_ms]['vars'] = sci_vars

            sci_vars_dict = cd.initialize_empty_arrays(stream_sci_vars_dict, ms)
            print('\nAppending data from files: {}'.format(ms))
            fcount = 0
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

                        print(fcount, var, varD.shape, tD.shape, sh['values'].shape, sh['t'].shape)

                        # put deployments time series together

                        # np.append(sh['t'], tD)
                        # np.append(sh['values'], varD)
                        sh['t'] = np.append(sh['t'], tD) #np.concatenate((sh['t'], tD), axis=0)

                        if fcount == 0:
                            if varD.ndim > 1:
                                sh['values'] = np.zeros(shape=[varD.shape[0], varD.shape[1]])
                                sh['values'][:] = varD
                            else:
                                sh['values'] = np.append(sh['values'], varD)
                        else:
                            if varD.ndim > 1:
                                sh['values'] = np.vstack([sh['values'], varD])
                            else:
                                sh['values'] = np.append(sh['values'], varD)

                        print(fcount, var, sh['values'].shape, sh['t'].shape)
                fcount += 1


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

                    if len(y) > 0:
                        if m == 'common_stream_placeholder':
                            sname = '-'.join((r, sv))
                        else:
                            sname = '-'.join((r, m, sv))

                        col = [str(x) for x in list(range(1, y.shape[1] + 1))]
                        col.insert(0, 'time')

                        groups, d_groups = gt.group_by_time_frequency(x, y, col, 'A')

                        time_ind = list(range(1, len(d_groups.columns), len(col)))
                        data_ind0 = list(range(2, len(d_groups.columns), len(col)))
                        data_ind1 = list(range(len(col), len(d_groups.columns), len(col)))
                        data_ind1.insert(len(groups)-1, len(d_groups.columns))

                        save_file = os.path.join(save_dir, sname)
                        fig, ax = pyplot.subplots(nrows=len(groups)+1, ncols=1)

                        for group in range(len(groups)):
                            nan_ind = d_groups[time_ind[group]].notnull()
                            xtime = d_groups[time_ind[group]][nan_ind]
                            n_year = xtime[0].year
                            ycol = list(range(data_ind0[group], data_ind1[group] + 1))

                            ydata = d_groups[ycol][nan_ind]
                            ydata = ydata.set_index(xtime)

                            if ydata.ndim > 1:
                                print('D > 1')
                                b = fsplt.split_by_timegap(ydata, 86400)

                                if b:
                                    print('b exists')
                                    for ib in (range(len(b))):
                                        iydata = b[ib][b[ib].columns.values[0:-2]]
                                        Y = iydata.columns.values
                                        X = iydata.index.values
                                        Z = iydata.values
                                        if Z.shape[0] == 1:
                                            X = np.repeat(X[0], len(Y), axis=0)
                                            df = pd.DataFrame(dict(a=list(X), b=list(Y), c=list(Z[0])))
                                            ax[group].scatter(df['a'].values, df['b'].values, c=df['c'].values, cmap=pyplot.cm.jet, s=1)
                                        else:
                                            Z = Z.T
                                            x, y = np.meshgrid(X, Y)
                                            ax[group].contourf(x, y, Z, alpha=0.7, cmap=pyplot.cm.jet)

                                else:
                                    print('no b')
                                    iydata = ydata[ydata.columns.values[0:-2]]
                                    Y = iydata.columns.values
                                    X = iydata.index.values
                                    Z = iydata.values
                                    Z = Z.T
                                    x, y = np.meshgrid(X, Y)
                                    im = ax[group].contourf(x, y, Z, alpha=0.7, cmap=pyplot.cm.jet)

                            else:
                                print('D = 1')
                                ydata.plot(ax=ax[group], linestyle='None', marker='.', markersize=0.5,
                                           color=colors[group])

                                ax[group].legend().set_visible(False)

                                # plot Mean and Standard deviation
                                ma = serie_n.rolling('86400s').mean()
                                mstd = serie_n.rolling('86400s').std()

                                ax[group].plot(ma.index, ma[col_name].values, 'k', linewidth=0.15)
                                ax[group].fill_between(mstd.index, ma[col_name].values - 2 * mstd[col_name].values,
                                                    ma[col_name].values + 2 * mstd[col_name].values,
                                                    color='b', alpha=0.2)

                            # prepare the time axis parameters
                            datemin = datetime.date(n_year, 1, 1)
                            datemax = datetime.date(n_year, 12, 31)
                            ax[group].set_xlim(datemin, datemax)
                            xlocator = mdates.MonthLocator()  # every month
                            myFmt = mdates.DateFormatter('%m')
                            ax[group].xaxis.set_minor_locator(xlocator)
                            ax[group].xaxis.set_major_formatter(myFmt)

                            # prepare the time axis parameters
                            ylocator = MaxNLocator(prune='both', nbins=3)
                            ax[group].yaxis.set_major_locator(ylocator)

                            # format figure
                            ax[group].tick_params(axis='both', color='r', labelsize=7, labelcolor='m')
                            if group == 0:
                                ax[group].set_title(sv + '( ' + sv_units, fontsize=8)
                            if group < len(groups) - 1:
                                ax[group].tick_params(which='both', pad=0.1, length=1, labelbottom=False)
                                ax[group].set_xlabel(' ')
                            else:
                                ax[group].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                                      pad=0.1, length=1, rotation=0)
                                ax[group].set_xlabel('Months', rotation=0, fontsize=8, color='b')
                                # pyplot.colorbar(im, extend='both', shrink=0.9, orientation="horizontal",
                                #                 pad=-0.5, ax=ax[group])


                            ax[group].set_ylabel(n_year, rotation=0, fontsize=8, color='b', labelpad=20)
                            ax[group].yaxis.set_label_position("right")

                            ymin, ymax = ax[group].get_ylim()
                            dep = 1
                            for etimes in end_times:
                                # if etimes < x_time[len(x_time)-1]:
                                ax[group].axvline(x=etimes, color='b', linestyle='--', linewidth=.6)
                                ax[group].text(etimes, ymin, 'End' + str(dep), fontsize=6, style='italic',
                                            bbox=dict(boxstyle='round',
                                                      ec=(0., 0.5, 0.5),
                                                      fc=(1., 1., 1.))
                                            )
                                dep += 1

                        fig.colorbar(im, cax=ax[len(groups)], orientation="horizontal")

                        fig.savefig(str(save_file), dpi=150)
                        pyplot.close()



if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'
    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160622-CE06ISSM-RID16-06-PHSEND000-recovered_host-phsen_abcdef_dcl_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160645-CE06ISSM-RID16-06-PHSEND000-recovered_inst-phsen_abcdef_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160700-CE06ISSM-RID16-06-PHSEND000-telemetered-phsen_abcdef_dcl_instrument/catalog.html'
                ]
main(sDir, url_list)

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160513-CE06ISSM-RID16-05-PCO2WB000-recovered_host-pco2w_abc_dcl_instrument_blank_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160606-CE06ISSM-RID16-05-PCO2WB000-telemetered-pco2w_abc_dcl_instrument_blank/catalog.html'
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160530-CE06ISSM-RID16-05-PCO2WB000-recovered_host-pco2w_abc_dcl_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160550-CE06ISSM-RID16-05-PCO2WB000-telemetered-pco2w_abc_dcl_instrument/catalog.html'

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T155131-CE06ISSM-RID16-02-FLORTD000-recovered_host-flort_sample/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T155144-CE06ISSM-RID16-02-FLORTD000-telemetered-flort_sample/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160252-CE06ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160441-CE06ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160458-CE06ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160715-CE06ISSM-RID16-07-NUTNRB000-recovered_host-nutnr_b_dcl_conc_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160735-CE06ISSM-RID16-07-NUTNRB000-recovered_host-nutnr_b_dcl_dark_conc_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160748-CE06ISSM-RID16-07-NUTNRB000-recovered_inst-nutnr_b_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160803-CE06ISSM-RID16-07-NUTNRB000-telemetered-nutnr_b_dcl_conc_instrument/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160819-CE06ISSM-RID16-07-NUTNRB000-telemetered-nutnr_b_dcl_dark_conc_instrument/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20190114T160837-CE06ISSM-RID16-07-NUTNRB000-telemetered-nutnr_b_dcl_full_instrument/catalog.html']

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T161639-CE09OSSM-RID27-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T161513-CE09OSSM-RID27-03-CTDBPC000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T161501-CE09OSSM-RID27-03-CTDBPC000-recovered_host-ctdbp_cdef_dcl_instrument_recovered/catalog.html']


# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154700-CE06ISSM-RID16-03-CTDBPC000-recovered_host-ctdbp_cdef_dcl_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154713-CE06ISSM-RID16-03-CTDBPC000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181217T154849-CE06ISSM-RID16-03-CTDBPC000-telemetered-ctdbp_cdef_dcl_instrument/catalog.html']

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163408-CE06ISSM-RID16-03-DOSTAD000-recovered_host-dosta_abcdjm_ctdbp_dcl_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163419-CE06ISSM-RID16-03-DOSTAD000-recovered_inst-dosta_abcdjm_ctdbp_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163558-CE06ISSM-RID16-03-DOSTAD000-telemetered-dosta_abcdjm_ctdbp_dcl_instrument/catalog.html']

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163845-CE09OSSM-RID27-04-DOSTAD000-recovered_host-dosta_abcdjm_dcl_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163907-CE09OSSM-RID27-04-DOSTAD000-telemetered-dosta_abcdjm_dcl_instrument/catalog.html']

# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163751-CE09OSPM-WFP01-02-DOFSTK000-recovered_wfp-dofst_k_wfp_instrument_recovered/catalog.html',
# 'https://opendap.oceanobservatories.org/thredds/catalog/ooi/leila.ocean@gmail.com/20181211T163824-CE09OSPM-WFP01-02-DOFSTK000-telemetered-dofst_k_wfp_instrument/catalog.html',