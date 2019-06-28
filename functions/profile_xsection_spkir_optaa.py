#!/usr/bin/env python
"""
Created on June 18 2019 by Lori Garzio

@brief: This script is used to plot profile and xsection plots for SPKIR on profilers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions.common as cf
import functions.plotting as pf
import functions.group_by_timerange as gt


def pf_xs_optaa(ds, sv, time1, y1, ds_lat1, ds_lon1, zcell_size, inpercentile, save_dir_profile, save_dir_xsection,
                deployment, press, y_units, n_std, zdbar):
    r = '-'.join((ds.subsite, ds.node, ds.sensor))
    fv = ds[sv]._FillValue
    sv_units = ds[sv].units
    if sv == 'optical_absorption':
        wv = ds['wavelength_a'].values
    else:
        wv = ds['wavelength_c'].values
    wavelengths = []
    iwavelengths = []
    for iw in range(len(wv)):
        if (wv[iw] > 671.) and (wv[iw] < 679.):
            wavelengths.append(wv[iw])
            iwavelengths.append(iw)
    zz = ds[sv].values.T
    for i in range(len(iwavelengths)):
        z1 = zz[i]

        # Check if the array is all NaNs
        if sum(np.isnan(z1)) == len(z1):
            print('Array of all NaNs - skipping plot.')
            continue

        # Check if the array is all fill values
        elif len(z1[z1 != fv]) == 0:
            print('Array of all fill values - skipping plot.')
            continue

        else:
            # remove unreasonable pressure data (e.g. for surface piercing profilers)
            if zdbar:
                po_ind = (0 < y1) & (y1 < zdbar)
                tm = time1[po_ind]
                y = y1[po_ind]
                z = z1[po_ind]
                ds_lat = ds_lat1[po_ind]
                ds_lon = ds_lon1[po_ind]
            else:
                tm = time1
                y = y1
                z = z1
                ds_lat = ds_lat1
                ds_lon = ds_lon1

            # reject erroneous data
            dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max, lat, lon = \
                cf.reject_erroneous_data(r, sv, tm, y, z, fv, ds_lat, ds_lon)

            if 'CTD' in r:
                ind = zpressure > 0.0
            else:
                ind = ndata > 0.0

            lenzero = np.sum(~ind)
            dtime = dtime[ind]
            zpressure = zpressure[ind]
            ndata = ndata[ind]
            if ds_lat is not None and ds_lon is not None:
                lat = lat[ind]
                lon = lon[ind]
            else:
                lat = None
                lon = None

            if len(dtime) > 0:
                # reject time range from data portal file export
                t_portal, z_portal, y_portal, lat_portal, lon_portal = \
                    cf.reject_timestamps_dataportal(ds.subsite, r, dtime, zpressure, ndata, lat, lon)

                print('removed {} data points using visual inspection of data'.format(
                    len(ndata) - len(z_portal)))

                # create data groups
                columns = ['tsec', 'dbar', str(sv)]
                min_r = int(round(np.nanmin(y_portal) - zcell_size))
                max_r = int(round(np.nanmax(y_portal) + zcell_size))
                ranges = list(range(min_r, max_r, zcell_size))

                groups, d_groups = gt.group_by_depth_range(t_portal, y_portal, z_portal, columns, ranges)

                if 'scatter' in sv:
                    n_std = None  # to use percentile
                else:
                    n_std = n_std

                #  get percentile analysis for printing on the profile plot
                y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex = cf.reject_timestamps_in_groups(
                    groups, d_groups, n_std, inpercentile)

            """
            Plot all data
            """
            if len(tm) > 0:
                method = ds.collection_method
                cf.create_dir(save_dir_profile)
                cf.create_dir(save_dir_xsection)
                plt_wavelength = ''.join((str(wavelengths[i]), 'nm'))
                sname = '-'.join((r, method, sv, ''.join((str(int(wavelengths[i])), 'nm'))))
                sfileall = '_'.join(('all_data', sname, pd.to_datetime(tm.min()).strftime('%Y%m%d')))
                tm0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
                tm1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, r, method)) + '\n' + tm0 + ' to ' + tm1
                title2 = 'Wavelength = {}'.format(plt_wavelength)
                title = title + '\n' + title2

                '''
                profile plot
                '''
                xlabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"
                clabel = 'Time'

                fig, ax = pf.plot_profiles(z1, y1, time1, ylabel, xlabel, clabel, stdev=None)

                ax.set_title(title, fontsize=9)
                fig.tight_layout()
                pf.save_fig(save_dir_profile, sfileall)

                '''
                xsection plot
                '''
                clabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"

                fig, ax, bar = pf.plot_xsection(ds.subsite, time1, y1, z1, clabel, ylabel, t_eng=None,
                                                m_water_depth=None, inpercentile=None, stdev=None)

                if fig:
                    ax.set_title(title, fontsize=9)
                    fig.tight_layout()
                    pf.save_fig(save_dir_xsection, sfileall)

            """
            Plot cleaned-up data
            """
            if len(dtime) > 0:
                sfile = '_'.join(('rm_erroneous_data', sname, pd.to_datetime(t_portal.min()).strftime('%Y%m%d')))
                t0 = pd.to_datetime(t_portal.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(t_portal.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, r, method)) + '\n' + t0 + ' to ' + t1
                title = title + '\n' + title2

                '''
                profile plot
                '''
                xlabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"
                clabel = 'Time'

                fig, ax = pf.plot_profiles(z_portal, y_portal, t_portal, ylabel, xlabel, clabel, stdev=None)

                ax.set_title(title, fontsize=9)
                ax.plot(n_avg, y_avg, '-k')
                ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                leg_text = (
                    'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                    '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                    '\nexcluded {} suspect data points when inspected visually'.format(
                        len(ndata) - len(z_portal)) +
                    '\n(black) data average in {} dbar segments'.format(zcell_size) +
                    '\n(magenta) {} percentile envelope in {} dbar segments'.format(
                        int(100 - inpercentile * 2), zcell_size),)
                ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                fig.tight_layout()
                pf.save_fig(save_dir_profile, sfile)

                '''
                xsection plot
                '''
                clabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"

                # plot non-erroneous data
                fig, ax, bar = pf.plot_xsection(ds.subsite, t_portal, y_portal, z_portal, clabel, ylabel,
                                                t_eng=None, m_water_depth=None, inpercentile=None,
                                                stdev=None)

                ax.set_title(title, fontsize=9)
                leg_text = (
                    'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                    '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                    '\nexcluded {} suspect data points when inspected visually'.format(
                        len(ndata) - len(z_portal)),
                )
                ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                fig.tight_layout()
                pf.save_fig(save_dir_xsection, sfile)


def pf_xs_spkir(ds, sv, time1, y1, ds_lat1, ds_lon1, zcell_size, inpercentile, save_dir_profile, save_dir_xsection,
                deployment, press, y_units, n_std, zdbar):
    r = '-'.join((ds.subsite, ds.node, ds.sensor))
    fv = ds[sv]._FillValue
    sv_units = ds[sv].units
    wavelengths = ['412nm', '443nm', '490nm', '510nm', '555nm', '620nm', '683nm']
    zz = ds[sv].values.T
    for i in range(len(wavelengths)):
        z1 = zz[i]
        # Check if the array is all NaNs
        if sum(np.isnan(z1)) == len(z1):
            print('Array of all NaNs - skipping plot.')
            continue

        # Check if the array is all fill values
        elif len(z1[z1 != fv]) == 0:
            print('Array of all fill values - skipping plot.')
            continue

        else:
            # remove unreasonable pressure data (e.g. for surface piercing profilers)
            if zdbar:
                po_ind = (0 < y1) & (y1 < zdbar)
                tm = time1[po_ind]
                y = y1[po_ind]
                z = z1[po_ind]
                ds_lat = ds_lat1[po_ind]
                ds_lon = ds_lon1[po_ind]
            else:
                tm = time1
                y = y1
                z = z1
                ds_lat = ds_lat1
                ds_lon = ds_lon1

            # reject erroneous data
            dtime, zpressure, ndata, lenfv, lennan, lenev, lengr, global_min, global_max, lat, lon = \
                cf.reject_erroneous_data(r, sv, tm, y, z, fv, ds_lat, ds_lon)

            if 'CTD' in r:
                ind = zpressure > 0.0
            else:
                ind = ndata > 0.0

            lenzero = np.sum(~ind)
            dtime = dtime[ind]
            zpressure = zpressure[ind]
            ndata = ndata[ind]
            if ds_lat is not None and ds_lon is not None:
                lat = lat[ind]
                lon = lon[ind]
            else:
                lat = None
                lon = None

            if len(dtime) > 0:
                # reject time range from data portal file export
                t_portal, z_portal, y_portal, lat_portal, lon_portal = \
                    cf.reject_timestamps_dataportal(ds.subsite, r, dtime, zpressure, ndata, lat, lon)

                print('removed {} data points using visual inspection of data'.format(
                    len(ndata) - len(z_portal)))

                # create data groups
                columns = ['tsec', 'dbar', str(sv)]
                min_r = int(round(np.nanmin(y_portal) - zcell_size))
                max_r = int(round(np.nanmax(y_portal) + zcell_size))
                ranges = list(range(min_r, max_r, zcell_size))

                groups, d_groups = gt.group_by_depth_range(t_portal, y_portal, z_portal, columns, ranges)

                if 'scatter' in sv:
                    n_std = None  # to use percentile
                else:
                    n_std = n_std

                #  get percentile analysis for printing on the profile plot
                y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_ex = cf.reject_timestamps_in_groups(
                    groups, d_groups, n_std, inpercentile)

            """
            Plot all data
            """
            if len(tm) > 0:
                method = ds.collection_method
                cf.create_dir(save_dir_profile)
                cf.create_dir(save_dir_xsection)
                plt_wavelength = wavelengths[i]
                sname = '-'.join((r, method, sv, plt_wavelength))
                sfileall = '_'.join(('all_data', sname, pd.to_datetime(tm.min()).strftime('%Y%m%d')))
                tm0 = pd.to_datetime(tm.min()).strftime('%Y-%m-%dT%H:%M:%S')
                tm1 = pd.to_datetime(tm.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, r, method)) + '\n' + tm0 + ' to ' + tm1
                title2 = 'Wavelength = {}'.format(plt_wavelength)
                title = title + '\n' + title2

                '''
                profile plot
                '''
                xlabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"
                clabel = 'Time'

                fig, ax = pf.plot_profiles(z1, y1, time1, ylabel, xlabel, clabel, stdev=None)

                ax.set_title(title, fontsize=9)
                fig.tight_layout()
                pf.save_fig(save_dir_profile, sfileall)

                '''
                xsection plot
                '''
                clabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"

                fig, ax, bar = pf.plot_xsection(ds.subsite, time1, y1, z1, clabel, ylabel, t_eng=None,
                                                m_water_depth=None, inpercentile=None, stdev=None)

                if fig:
                    ax.set_title(title, fontsize=9)
                    fig.tight_layout()
                    pf.save_fig(save_dir_xsection, sfileall)

            """
            Plot cleaned-up data
            """
            if len(dtime) > 0:
                sfile = '_'.join(('rm_erroneous_data', sname, pd.to_datetime(t_portal.min()).strftime('%Y%m%d')))
                t0 = pd.to_datetime(t_portal.min()).strftime('%Y-%m-%dT%H:%M:%S')
                t1 = pd.to_datetime(t_portal.max()).strftime('%Y-%m-%dT%H:%M:%S')
                title = ' '.join((deployment, r, method)) + '\n' + t0 + ' to ' + t1
                title = title + '\n' + title2

                '''
                profile plot
                '''
                xlabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"
                clabel = 'Time'

                fig, ax = pf.plot_profiles(z_portal, y_portal, t_portal, ylabel, xlabel, clabel, stdev=None)

                ax.set_title(title, fontsize=9)
                ax.plot(n_avg, y_avg, '-k')
                ax.fill_betweenx(y_avg, n0_std, n1_std, color='m', alpha=0.2)
                leg_text = (
                    'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                    '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                    '\nexcluded {} suspect data points when inspected visually'.format(
                        len(ndata) - len(z_portal)) +
                    '\n(black) data average in {} dbar segments'.format(zcell_size) +
                    '\n(magenta) {} percentile envelope in {} dbar segments'.format(
                        int(100 - inpercentile * 2), zcell_size),)
                ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                fig.tight_layout()
                pf.save_fig(save_dir_profile, sfile)

                '''
                xsection plot
                '''
                clabel = sv + " (" + sv_units + ")"
                ylabel = press[0] + " (" + y_units[0] + ")"

                # plot non-erroneous data
                fig, ax, bar = pf.plot_xsection(ds.subsite, t_portal, y_portal, z_portal, clabel, ylabel,
                                                t_eng=None, m_water_depth=None, inpercentile=None,
                                                stdev=None)

                ax.set_title(title, fontsize=9)
                leg_text = (
                    'removed {} fill values, {} NaNs, {} Extreme Values (1e7), {} Global ranges [{} - {}], '
                    '{} zeros'.format(lenfv, lennan, lenev, lengr, global_min, global_max, lenzero) +
                    '\nexcluded {} suspect data points when inspected visually'.format(
                        len(ndata) - len(z_portal)),
                )
                ax.legend(leg_text, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=6)
                fig.tight_layout()
                pf.save_fig(save_dir_xsection, sfile)
