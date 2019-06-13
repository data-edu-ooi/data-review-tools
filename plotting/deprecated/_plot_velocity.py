
import os
import itertools
import pandas as pd
import xarray as xr
import numpy as np
import functions.common as cf
import matplotlib
from matplotlib import pyplot


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
            print(u)
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

        save_dir = os.path.join(sDir, array, subsite, r, 'preferred_method_plots')
        cf.create_dir(save_dir)



        # get the preferred stream information
        fig, ax = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        fig0, ax0 = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig0.tight_layout()

        fig1, ax1 = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig1.tight_layout()

        fig2, ax2 = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig2.tight_layout()

        fig3, ax3 = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig3.tight_layout()

        fig4, ax4 = pyplot.subplots(nrows=len(fdatasets), ncols=1, sharey=True)
        fig4.tight_layout()

        for ii in range(len(fdatasets)):
            print('\n', fdatasets[ii])
            deployment = fdatasets[ii].split('/')[-1].split('_')[0].split('deployment')[-1]
            deployment = int(deployment)

            ds = xr.open_dataset(fdatasets[ii], mask_and_scale=False)
            time = ds['time'].values
            sci_var = cf.return_science_vars(ds.stream)

            # Plot pressure
            z_name = [z_var for z_var in sci_var if 'pressure' in z_var]
            z = ds[z_name[0]].values
            z_unit = ds[z_name[0]].units

            ax1[ii].plot(time, z, 'b-', linestyle='--', linewidth=.6, label='V')
            ax1[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            ax1[ii].yaxis.set_label_position("right")
            ax1[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                pad=0.1, length=1, rotation=0)
            if ii < len(fdatasets) - 1:
                ax1[ii].set_xlabel(' ')
            else:
                ax1[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')

            if ii == 0:
                ax1[ii].set_title(r + ' - Pressure ' + z_unit, fontsize=8)

            sfile = 'pressure_plots'
            save_file = os.path.join(save_dir, sfile)
            fig1.savefig(str(save_file), dpi=150)

            # non science veriable
            # According to VELPT manufacturer, data are suspect when this instrument is tilted more than 20 degrees
            # redmine ticket: Marine Hardware #12960

            roll = ds['roll_decidegree'].values
            roll_unit = ds['roll_decidegree'].units
            pitch = ds['pitch_decidegree'].values
            pitch_units = ds['pitch_decidegree'].units
            headng = ds['heading_decidegree'].values
            headng_units = ds['heading_decidegree'].values

            tilt_ind = np.logical_or(pitch > 200, roll > 200)
            pitch_fit = pitch[tilt_ind]
            roll_fit = roll[tilt_ind]

            # plot roll
            ax2[ii].plot(time, roll, 'b-', linestyle='--', linewidth=.6, label='Roll')
            ax2[ii].plot(time[tilt_ind], roll_fit, 'g.', linestyle='None', marker='.', markersize=0.5, label='Roll < 200')
            ax2[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            ax2[ii].yaxis.set_label_position("right")
            ax2[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                pad=0.1, length=1, rotation=0)
            if ii < len(fdatasets) - 1:
                ax2[ii].set_xlabel(' ')
            else:
                ax2[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')

            if ii == 0:
                ax2[ii].set_title(r + ' - Roll ' + roll_unit, fontsize=8)
                leg2 = ax2[ii].legend(fontsize=6, bbox_to_anchor=(0., 0.80, 1., .102), loc=3,
                              ncol=3, mode="expand", borderaxespad=0.)
                leg2._drawFrame = False

            sfile = 'roll_plots'
            save_file = os.path.join(save_dir, sfile)
            fig2.savefig(str(save_file), dpi=150)

            # plot pitch
            ax3[ii].plot(time, pitch, 'b-', linestyle='--', linewidth=.6, label='Roll')
            ax3[ii].plot(time[tilt_ind], pitch_fit, 'g.', linestyle='None', marker='.', markersize=0.5, label='Roll < 200')
            ax3[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            ax3[ii].yaxis.set_label_position("right")
            ax3[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                pad=0.1, length=1, rotation=0)
            if ii < len(fdatasets) - 1:
                ax3[ii].set_xlabel(' ')
            else:
                ax3[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')

            if ii == 0:
                ax3[ii].set_title(r + ' - Pitch ' + roll_unit, fontsize=8)
                leg3 = ax2[ii].legend(fontsize=6, bbox_to_anchor=(0., 0.80, 1., .102), loc=3,
                              ncol=3, mode="expand", borderaxespad=0.)
                leg3._drawFrame = False

            sfile = 'pitch_plots'
            save_file = os.path.join(save_dir, sfile)
            fig3.savefig(str(save_file), dpi=150)

            # plot heading
            ax4[ii].plot(time, headng, 'b-', linestyle='None', marker='.', markersize=0.5, label='Roll')
            ax4[ii].plot(time[tilt_ind], headng[tilt_ind], 'g.', linestyle='None', marker='.', markersize=0.5, label='Roll < 200')
            ax4[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            ax4[ii].yaxis.set_label_position("right")
            ax4[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
                                pad=0.1, length=1, rotation=0)
            if ii < len(fdatasets) - 1:
                ax4[ii].set_xlabel(' ')
            else:
                ax4[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')

            if ii == 0:
                ax4[ii].set_title(r + ' - Heading ' + roll_unit, fontsize=8)
                leg4 = ax2[ii].legend(fontsize=6, bbox_to_anchor=(0., 0.80, 1., .102), loc=3,
                              ncol=3, mode="expand", borderaxespad=0.)
                leg4._drawFrame = False

            sfile = 'heading_plots'
            save_file = os.path.join(save_dir, sfile)
            fig4.savefig(str(save_file), dpi=150)


            # velocity variable
            u_name = [u_var for u_var in sci_var if 'eastward_velocity' in u_var]
            v_name = [v_var for v_var in sci_var if 'northward_velocity' in v_var]
            w_name = [w_var for w_var in sci_var if 'upward_velocity' in w_var]

            w = ds[w_name[0]].values
            w_unit = ds[w_name[0]].units
            u = ds[u_name[0]].values
            v = ds[v_name[0]].values
            uv_magnitude = np.sqrt(u ** 2 + v ** 2)
            uv_maxmag = max(uv_magnitude)


            # 1D Quiver plot
            ax[ii].quiver(time, 0, u, v,
                              color = 'r',
                              units = 'y',
                              scale_units = 'y',
                              scale = 1,
                              headlength = 1,
                              headaxislength = 1,
                              width = 0.004,
                              alpha = 0.5
                              )

            u_fit = u[tilt_ind]
            v_fit = v[tilt_ind]
            ax[ii].quiver(time[tilt_ind], 0, u_fit, v_fit,
                              color = 'b',
                              units = 'y',
                              scale_units = 'y',
                              scale = 1,
                              headlength = 1,
                              headaxislength = 1,
                              width = 0.004,
                              alpha = 0.5
                              )
            percent_bad = round(((len(u)-len(u_fit))/len(u))* 100, 2)
            print(len(u_fit), len(u), percent_bad)
            ax[ii].text(time[-1], 0, ' ' + str(percent_bad) + '%', fontsize=5, style='italic', color='blue')

            ax[ii].set_ylim(-uv_maxmag, uv_maxmag)
            ax[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            ax[ii].yaxis.set_label_position("right")
            ax[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m', pad=0.1, length=1, rotation=0)
            if ii < len(fdatasets) - 1:
                ax[ii].set_xlabel(' ')
            else:
                ax[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')

            if ii == 0:
                ax[ii].set_title(r + ' - Current Velocity ' + w_unit + '\n' +' Currents in blue when pitch or roll are > 20 degrees', fontsize=8)

            # ax[ii].text(time[0], uv_magnitude- 0.05, 'mim: ' + str(round(min(uv_magnitude),3)) + ' , max: ' + str(round(max(uv_magnitude),3)), fontsize=8)


            sfile = 'current_plot'
            save_file = os.path.join(save_dir, sfile)
            fig.savefig(str(save_file ), dpi=150, bbox_inches='tight')



            # Plot u and v components

            # ax0[ii].plot(time, v, 'b-', linestyle='--', linewidth=.6, label='V')
            # ax0[ii].plot(time, u, 'g-', linestyle='--', linewidth=.6, label='U')
            # ax0[ii].plot(time, w, 'r-', linestyle='--', linewidth=.6, label='W')
            #
            # ax0[ii].set_ylabel(str(deployment), rotation=0, fontsize=8, color='b', labelpad=11)
            # ax0[ii].yaxis.set_label_position("right")
            # ax0[ii].tick_params(which='both', color='r', labelsize=7, labelcolor='m',
            #                    pad=0.1, length=1, rotation=0)
            #
            # if ii < len(fdatasets) - 1:
            #     ax0[ii].set_xlabel(' ')
            # else:
            #     ax0[ii].set_xlabel('Time', rotation=0, fontsize=8, color='b')
            #
            #
            # # set title
            # if ii == 0:
            #     ax0[ii].set_title(r + ' - Velocity Components' + w_unit, fontsize=8)
            #     # Set legend location - See: http://matplotlib.org/users/legend_guide.html#legend-location
            #     leg0 = ax0[ii].legend(fontsize=6, bbox_to_anchor=(0., 0.80, 1., .102), loc=3,
            #                   ncol=3, mode="expand", borderaxespad=0.)
            #     leg0._drawFrame = False
            #
            # sfile = 'uv_plots'
            # save_file = os.path.join(save_dir, sfile)
            # fig0.savefig(str(save_file ), dpi=150)


if __name__ == '__main__':
    pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
    preferred_only = 'yes'
    sDir = '/Users/leila/Documents/NSFEduSupport/review/figures'

    url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190102T153947-CE01ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190102T153934-CE01ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
                'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190102T153921-CE01ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_dcl_instrument_recovered/catalog.html']

    # url_list = ['https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191340-CE06ISSM-RID16-04-VELPTA000-telemetered-velpt_ab_dcl_instrument/catalog.html',
    #             'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191211-CE06ISSM-RID16-04-VELPTA000-recovered_inst-velpt_ab_instrument_recovered/catalog.html',
    #             'https://opendap.oceanobservatories.org/thredds/catalog/ooi/lgarzio@marine.rutgers.edu/20190111T191157-CE06ISSM-RID16-04-VELPTA000-recovered_host-velpt_ab_instrument_recovered/catalog.html']

main(sDir, url_list, preferred_only)