#!/usr/bin/env python
import os
import itertools
import pandas as pd
import xarray as xr
import numpy as np
import functions.common as cf
import matplotlib
from matplotlib import pyplot
import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import (YEARLY, DateFormatter, rrulewrapper, RRuleLocator, drange)
from matplotlib.ticker import MaxNLocator

def reject_err_data_1_dims(y, y_fill, r, sv, n=None):
    n_nan = np.sum(np.isnan(y)) # count nans in data
    n_nan = n_nan.item()
    y = np.where(y != y_fill, y, np.nan) # replace fill_values by nans in data
    y = np.where(y != -9999, y, np.nan) # replace -9999 by nans in data
    n_fv = np.sum(np.isnan(y)) - n_nan# re-count nans in data
    n_fv = n_fv.item()
    y = np.where(y > -1e10, y, np.nan) # replace extreme values by nans in data
    y = np.where(y < 1e10, y, np.nan)
    n_ev = np.sum(np.isnan(y)) - n_fv - n_nan # re-count nans in data
    n_ev = n_ev.item()

    g_min, g_max = cf.get_global_ranges(r, sv) # get global ranges:
    if g_min and g_max:
        y = np.where(y >= g_min, y, np.nan) # replace extreme values by nans in data
        y = np.where(y <= g_max, y, np.nan)
        n_grange = np.sum(np.isnan(y)) - n_ev - n_fv - n_nan # re-count nans in data
        n_grange = n_grange.item()
    else:
        n_grange = np.nan

    stdev = np.nanstd(y)
    if stdev > 0.0:
        y = np.where(abs(y - np.nanmean(y)) < n * stdev, y, np.nan) # replace 5 STD by nans in data
        n_std = np.sum(np.isnan(y)) - ~np.isnan(n_grange) - n_ev - n_fv - n_nan # re-count nans in data
        n_std = n_std.item()

    err_count = pd.DataFrame({'n_nan':[n_nan],
                             'n_fv':[n_fv],
                             'n_ev':[n_ev],
                             'n_grange':[n_grange],
                             'g_min':[g_min],
                             'g_max':[g_max],
                             'n_std':[n_std]}, index=[0])
    return  y, err_count


def get_variable_data(ds, var_list, keyword):
    var = [var for var in var_list if keyword in var]
    if len(var) == 1:
        var_id = var[0]
        var_data = ds[var_id].values
        var_unit = ds[var_id].units
        var_name = ds[var_id].long_name
        var_fill = ds[var_id]._FillValue
    else:
        print('more than one matching name exist: ', var)

    return var_id, var_data, var_unit, var_name, var_fill


def prepare_axis(css, fig, ax, time, pressure, data, r, deployment, ii, num_files, err_count=None):
    # prepare the time axis parameters
    datemin = np.datetime64(min(time), 'M')
    datemax = np.datetime64(max(time), 'M') + np.timedelta64(1, 'M')
    ax.set_xlim(datemin, datemax)
    xlocator = mdates.MonthLocator()  # every month
    myFmt = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_minor_locator(xlocator)
    ax.xaxis.set_major_formatter(myFmt)

    ax.invert_yaxis()

    # ax.set_ylabel(('Deployment' + str(deployment)), fontsize=7, color='b', labelpad=11)
    # ax.yaxis.set_label_position("right")
    ax.tick_params(which='both', labelsize=7, pad=0.1, length=1, rotation=0)
    ax.set_ylabel(pressure[0] + '('+ pressure[1]+ ')', color='b', fontsize=7, labelpad=11)
    ax.yaxis.set_label_position("left")
    if ii < num_files - 1:
        ax.set_xlabel(' ')
    else:
        ax.set_xlabel(('Time'+ '\n' + 'Deployment' + str(deployment) + '   ' + r), color='b', rotation=0, fontsize=7)
    if err_count is not None:
        title_i = ('removed: {} nans, {} fill values, {} extreme values, {} GR [{}, {}],' \
                  ' {} outliers +/- 5 SD'.format(err_count['n_nan'].values,
                                                 err_count['n_fv'].values,
                                                 err_count['n_ev'].values,
                                                 err_count['n_grange'].values,
                                                 err_count['g_min'].values,
                                                 err_count['g_max'].values,
                                                 err_count['n_std'].values))
    else:
        title_i = ''

    ax.set_title(title_i, size="x-small")

    bar = fig.colorbar(css, ax=ax, label= data[0] + ' (' + data[1] + ')', extend='both')
    bar.formatter.set_useOffset(False)
    bar.ax.tick_params(labelsize=8)


def plot_velocity_variables(r, fdatasets, num_plots, save_dir):

    fig0_0, ax0_0 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig0_0.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig0_0_file = 'v_plots'

    fig0_1, ax0_1 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig0_1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig0_1_file = 'u_plots'

    fig0_2, ax0_2 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig0_2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig0_2_file = 'w_plots'

    # fig1, ax1 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    # fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # fig1_file = 'Calculated_current_plots'


    fig2, ax2 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig2_file = 'roll_plots'

    fig3, ax3 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig3_file = 'pitch_plots'

    fig4, ax4 = pyplot.subplots(nrows=num_plots, ncols=1, sharey=True)
    fig4.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig4_file = 'calculated_current_magnitude_plots'

    if num_plots > len(fdatasets):
        for jj in range(len(fdatasets), num_plots, 1):
            ax0_0[jj].axis('off')
            ax0_1[jj].axis('off')
            ax0_2[jj].axis('off')
            # ax1[jj].axis('off')
            ax2[jj].axis('off')
            ax3[jj].axis('off')
            ax4[jj].axis('off')

    if len(fdatasets) == 1:
        ax0_0 = [ax0_0]
        ax0_1 = [ax0_1]
        ax0_2 = [ax0_2]
        # ax1 = [ax1]
        ax2 = [ax2]
        ax3 = [ax3]
        ax4 = [ax4]

    for ii in range(len(fdatasets)):

        print('\n', fdatasets[ii].split('/')[-1])
        deployment = fdatasets[ii].split('/')[-1].split('_')[0].split('deployment')[-1]
        deployment = int(deployment)

        fig2_file = fig2_file + '_deployment' +str(deployment)+ '_' + fdatasets[ii].split('/')[-1].split('_')[-1].split('.')[0]
        print(fig2_file)

        ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
        time = ds['time'].values

        '''
        variable list
        '''
        var_list = cf.notin_list(ds.data_vars.keys(), ['time', '_qc_'])

        z_data, z_unit, z_name, z_fill = cf.add_pressure_to_dictionary_of_sci_vars(ds)
        z_data, err_count_z = reject_err_data_1_dims(z_data, z_fill[0], r, z_name[0], n=5)

        w_id, w_data, w_unit, w_name, w_fill = get_variable_data(ds, var_list, 'upward_velocity')
        w_data, err_count_w = reject_err_data_1_dims(w_data, w_fill, r, w_id, n=5)

        u_id, u_data, u_unit, u_name, u_fill = get_variable_data(ds, var_list, 'eastward_velocity')
        u_data, err_count_u = reject_err_data_1_dims(u_data, u_fill, r, u_id, n=5)

        v_id, v_data, v_unit, v_name, v_fill = get_variable_data(ds, var_list, 'northward_velocity')
        v_data, err_count_v = reject_err_data_1_dims(v_data, v_fill, r, v_id, n=5)

        roll_id, roll_data, roll_unit, roll_name, roll_fill = get_variable_data(ds, var_list, 'roll')
        roll_data, err_count_roll = reject_err_data_1_dims(roll_data, roll_fill, r, roll_id, n=5)

        pitch_id, pitch_data, pitch_unit, pitch_name, pitch_fill = get_variable_data(ds, var_list, 'pitch')
        pitch_data, err_count_pitch = reject_err_data_1_dims(pitch_data, pitch_fill, r, pitch_id, n=5)

        '''
        2D Quiver plot
        '''
        # ax1[ii].quiver(time, z_data, u_data, v_data, color='b', units='y', scale_units='y', scale=1, headlength=1,
        #                headaxislength=1, width=0.004, alpha=0.5)
        # M = np.sqrt(u_data ** 2 + v_data ** 2)
        # Q = ax1[ii].quiver(time[::100], z_data[::100], u_data[::100], v_data[::100], M[::100],
        #                    units='y', pivot='tip', width=0.022, scale=1 / 0.15)
        # css = ax1[ii].quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E', coordinates='figure')
        # #
        # prepare_axis(css, fig1, ax1[ii],time, [z_name[0], z_unit[0]], ['Current Velocity', u_unit], r, deployment,  ii, len(fdatasets),
        #                    err_count=None)
        #
        # fig1_file = fig1_file + str(deployment)
        #
        '''
        plot roll
        '''
        css = ax2[ii].scatter(roll_data, z_data, c=time, s=2, edgecolor='None', cmap='RdGy')
        # css = ax2[ii].scatter(time, z_data, c=roll_data, cmap='RdGy', s=2, edgecolor='None')
        prepare_axis(css, fig2, ax2[ii], time, [z_name[0], z_unit[0]], [roll_name, roll_unit], r, deployment, ii,
                                                                            len(fdatasets), err_count=err_count_roll)
        fig2_file = fig2_file + str(deployment)
        #
    #     '''
    #     plot pitch
    #     '''
    #     css = ax3[ii].scatter(time, z_data, c=pitch_data, cmap='RdGy', s=2, edgecolor='None')
    #     prepare_axis(css, fig3, ax3[ii], time, [z_name[0], z_unit[0]], [pitch_name, pitch_unit], r, deployment, ii,
    #                                                                         len(fdatasets), err_count=err_count_pitch)
    #     fig3_file = fig3_file + str(deployment)
    #     '''
    #      plot current magnitude
    #     '''
    #     uv_magnitude = np.sqrt(u_data ** 2 + v_data ** 2)
    #     css = ax4[ii].scatter(time, z_data, c=uv_magnitude, cmap='PuBu', s=2, edgecolor='None')
    #     prepare_axis(css, fig4, ax4[ii], time, [z_name[0], z_unit[0]], ['[U,V] Current Velocity', u_unit], r, deployment, ii,
    #                                                                                     len(fdatasets), err_count=None)
    #     fig4_file = fig4_file + str(deployment)
    #
    #     '''
    #     Plot v component
    #     '''
    #     css = ax0_0[ii].scatter(time, z_data, c=v_data, cmap='RdBu', s=2, edgecolor='None')
    #     prepare_axis(css, fig0_0, ax0_0[ii], time, [z_name[0], z_unit[0]], ['V Components', v_unit], r, deployment, ii,
    #                                                                             len(fdatasets), err_count=err_count_v)
    #     fig0_0_file = fig0_0_file + str(deployment)
    #
    #     '''
    #     Plot u component
    #     '''
    #     css = ax0_1[ii].scatter(time, z_data, c=u_data, cmap='RdBu', s=2, edgecolor='None')
    #     prepare_axis(css, fig0_1, ax0_1[ii], time, [z_name[0], z_unit[0]], ['U Components', u_unit], r, deployment, ii,
    #                                                                             len(fdatasets), err_count=err_count_u)
    #
    #     fig0_1_file = fig0_1_file + str(deployment)
    #
    #     '''
    #     Plot w component
    #     '''
    #     css = ax0_2[ii].scatter(time, z_data, c=w_data, cmap='RdBu', s=2, edgecolor='None')
    #     prepare_axis(css, fig0_2, ax0_2[ii], time, [z_name[0], z_unit[0]], ['W Components', w_unit], r, deployment, ii,
    #                                                                             len(fdatasets), err_count=err_count_w)
    #
    #     fig0_2_file = fig0_2_file + str(deployment)
    #
    #
    # # save_file = os.path.join(save_dir, fig1_file)
    # # fig1.savefig(str(save_file), dpi=150, bbox_inches='tight')

    save_file = os.path.join(save_dir, fig2_file)
    fig2.savefig(str(save_file), dpi=150, bbox_inches='tight')

    # save_file = os.path.join(save_dir, fig3_file)
    # fig3.savefig(str(save_file), dpi=150, bbox_inches='tight')
    # #
    # save_file = os.path.join(save_dir, fig4_file)
    # fig4.savefig(str(save_file), dpi=150, bbox_inches='tight')
    #
    # save_file = os.path.join(save_dir, fig0_0_file)
    # fig0_0.savefig(str(save_file), dpi=150, bbox_inches='tight')
    # #
    # save_file = os.path.join(save_dir, fig0_1_file)
    # fig0_1.savefig(str(save_file), dpi=150, bbox_inches='tight')
    # #
    # save_file = os.path.join(save_dir, fig0_2_file)
    # fig0_2.savefig(str(save_file), dpi=150, bbox_inches='tight')

    pyplot.close(fig0_0)
    pyplot.close(fig0_1)
    pyplot.close(fig0_2)
    pyplot.close(fig2)
    pyplot.close(fig3)
    pyplot.close(fig4)
def main(sDir, url_list, preferred_only):
    rd_list = []
    for uu in url_list:
        elements = uu.split('/')[-2].split('-')
        rd = '-'.join((elements[1], elements[2], elements[3], elements[4]))
        ms = uu.split(rd + '-')[1].split('/')[0]
        if rd not in rd_list:
            rd_list.append(rd)

    for r in rd_list:
        print('\n{}'.format(r))
        subsite = r.split('-')[0]
        array = subsite[0:2]

        datasets = []
        for u in url_list:
            splitter = u.split('/')[-2].split('-')
            rd_check = '-'.join((splitter[1], splitter[2], splitter[3], splitter[4]))
            if rd_check == r:
                udatasets = cf.get_nc_urls([u])
                datasets.append(udatasets)

        datasets = list(itertools.chain(*datasets))

        if preferred_only == 'yes':

            ps_df, n_streams = cf.get_preferred_stream_info(r)

            fdatasets = []
            for index, row in ps_df.iterrows():
                for ii in range(n_streams):
                    try:
                        rms = '-'.join((r, row[ii]))
                    except TypeError:
                        continue
                    for dd in datasets:
                        spl = dd.split('/')[-2].split('-')
                        catalog_rms = '-'.join((spl[1], spl[2], spl[3], spl[4], spl[5], spl[6]))
                        fdeploy = dd.split('/')[-1].split('_')[0]
                        if rms == catalog_rms and fdeploy == row['deployment']:
                            fdatasets.append(dd)
        else:
            fdatasets = datasets

        main_sensor = r.split('-')[-1]
        fdatasets = cf.filter_collocated_instruments(main_sensor, fdatasets)
        num_data = len(fdatasets)
        save_dir = os.path.join(sDir, array, subsite, r, 'preferred_method_plots')
        cf.create_dir(save_dir)

        num_plots = 1
        if len(fdatasets) > num_plots:
            steps = list(range(num_plots, len(fdatasets)+num_plots, num_plots))
            for ii in steps:
                plot_velocity_variables(r, fdatasets[ii-num_plots:ii], num_plots, save_dir)
        else:
            plot_velocity_variables(r, fdatasets, num_plots, save_dir)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    preferred_only = 'yes'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'

    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021729-CE09OSPM-WFP01-01-VEL3DK000-recovered_wfp-vel3d_k_wfp_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20181213T021754-CE09OSPM-WFP01-01-VEL3DK000-telemetered-vel3d_k_wfp_stc_instrument/catalog.html']

main(sDir, url_list, preferred_only)