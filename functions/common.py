#! /usr/bin/env python
import os
import pandas as pd
import requests
import re
import itertools
import time
import xarray as xr
import numpy as np
import datetime as dt
from urllib.request import urlopen
import json
from geopy.distance import geodesic
import urllib3
import functions.plotting as pf
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def check_request_status(thredds_url):
    check_complete = thredds_url.replace('/catalog/', '/fileServer/')
    check_complete = check_complete.replace('/catalog.html', '/status.txt')
    session = requests.session()
    r = session.get(check_complete)
    while r.status_code != requests.codes.ok:
        print('Data request is still fulfilling. Trying again in 1 minute.')
        time.sleep(60)
        r = session.get(check_complete)
    print('Data request has fulfilled.')


def create_dir(new_dir):
    # Check if dir exists.. if it doesn't... create it.
    if not os.path.isdir(new_dir):
        try:
            os.makedirs(new_dir)
        except OSError:
            if os.path.exists(new_dir):
                pass
            else:
                raise


def deploy_location_check(refdes):
    # Calculate the distance in kilometers between an instrument's location (defined in asset management) and previous
    # deployment locations
    deploy_loc = {}
    dr_data = refdes_datareview_json(refdes)
    for i, d in enumerate(dr_data['instrument']['deployments']):
        deploy_loc[i] = {}
        deploy_loc[i]['deployment'] = d['deployment_number']
        deploy_loc[i]['lat'] = d['latitude']
        deploy_loc[i]['lon'] = d['longitude']

    # put info in a data frame
    df = pd.DataFrame.from_dict(deploy_loc, orient='index').sort_index()
    y = {}
    for i, k in df.iterrows():
        if i > 0:
            loc1 = [k['lat'], k['lon']]
            d1 = int(k['deployment'])
            for x in range(i):
                info0 = df.iloc[x]
                compare = 'diff_km_D{}_to_D{}'.format(d1, int(info0['deployment']))
                loc0 = [info0['lat'], info0['lon']]
                diff_loc = round(geodesic(loc0, loc1).kilometers, 4)
                y.update({compare: diff_loc})
    return y


def filter_collocated_instruments(main_sensor, datasets):
    # Remove collocated instruments from a list of datasets from THREDDS
    datasets_filtered = []
    for d in datasets:
        fname = d.split('/')[-1]
        if main_sensor in fname:
            datasets_filtered.append(d)
    return datasets_filtered


def filter_other_streams(r, stream_list, fdatasets):
    # Remove other streams from a list of datasets from THREDDS
    datasets_filtered = []
    for d in fdatasets:
        fname = d.split(r + '-')[-1].split('_2')[0]
        for s in stream_list:
            if s == fname:
                if d not in datasets_filtered:
                    datasets_filtered.append(d)


    return datasets_filtered


def get_deployment_information(data, deployment):
    d_info = [x for x in data['instrument']['deployments'] if x['deployment_number'] == deployment]
    if d_info:
        return d_info[0]
    else:
        return None

def get_url_content(url_address):
    # get content of a url in a json format
    r = requests.get(url_address)
    if r.status_code is not 200:
        print(r.reason)
        print('Problem wi chatth', url_address)
    else:
        url_content = r.json()
    return url_content

def get_global_ranges(refdes, variable, api_user=None, api_token=None):
    port = '12578'
    spl = refdes.split('-')
    base_url = '{}/qcparameters/inv/{}/{}/{}/'.format(port, spl[0], spl[1], '-'.join((spl[2], spl[3])))
    url = 'https://ooinet.oceanobservatories.org/api/m2m/{}'.format(base_url)
    if (api_user is None) or (api_token is None):
        r = requests.get(url, verify=False)
    else:
        r = requests.get(url, auth=(api_user, api_token), verify=False)

    if r.status_code is 200:
        if r.json():  # If r.json is not empty
            values = pd.io.json.json_normalize(r.json())
            t1 = values[values['qcParameterPK.streamParameter'] == variable]
            if not t1.empty:
                t2 = t1[t1['qcParameterPK.qcId'] == 'dataqc_globalrangetest_minmax']
                if not t2.empty:
                    global_min = float(t2[t2['qcParameterPK.parameter'] == 'dat_min'].iloc[0]['value'])
                    global_max = float(t2[t2['qcParameterPK.parameter'] == 'dat_max'].iloc[0]['value'])
                else:
                    global_min = None
                    global_max = None
            else:
                global_min = None
                global_max = None
        else:
            global_min = None
            global_max = None
    else:
        raise Exception('uFrame is not responding to request for global ranges. Try again later.')

    return [global_min, global_max]


def get_nc_urls(catalog_urls):
    """
    Return a list of urls to access netCDF files in THREDDS
    :param catalog_urls: List of THREDDS catalog urls
    :return: List of netCDF urls for data access
    """
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC'
    datasets = []
    for i in catalog_urls:
        # check that the request has fulfilled
        check_request_status(i)

        dataset = requests.get(i).text
        ii = re.findall(r'href=[\'"]?([^\'" >]+)', dataset)
        x = re.findall(r'(ooi/.*?.nc)', dataset)
        for i in x:
            if i.endswith('.nc') == False:
                x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        dataset = [os.path.join(tds_url, i) for i in x]
        datasets.append(dataset)
    datasets = list(itertools.chain(*datasets))
    return datasets


def get_preferred_stream_info(refdes):
    ps_link = 'https://raw.githubusercontent.com/ooi-data-lab/data-review-tools/master/data_review/output/{}/{}/{}-preferred_stream.json'.format(
        refdes.split('-')[0], refdes, refdes)
    pslnk = urlopen(ps_link)
    psl = json.loads(pslnk.read())
    ps_df = pd.DataFrame.from_dict(psl, orient='index')
    ps_df = ps_df.reset_index()
    ps_df.rename(columns={'index': 'deployment'}, inplace=True)
    ps_df.sort_values(by=['deployment'], inplace=True)
    n_streams = len(ps_df.columns) - 1

    return ps_df, n_streams


def nc_attributes(nc_file):
    """
    Return global information from a netCDF file
    :param nc_file: url for a netCDF file on the THREDDs server
    """
    with xr.open_dataset(nc_file) as ds:
        fname = nc_file.split('/')[-1].split('.nc')[0]
        subsite = ds.subsite
        node = ds.node
        sensor = ds.sensor
        refdes = '-'.join((subsite, node, sensor))
        method = ds.collection_method
        stream = ds.stream
        deployment = fname[0:14]

    return fname, subsite, refdes, method, stream, deployment


def refdes_datareview_json(refdes):
    # returns information about a reference designator from the Data Review Database
    url = 'http://datareview.marine.rutgers.edu/instruments/view/'
    ref_des_url = os.path.join(url, refdes)
    ref_des_url += '.json'
    r = requests.get(ref_des_url).json()
    return r


def reject_global_ranges(data, gmin, gmax):
    # Reject data outside of global ranges
    return (data >= gmin) & (data <= gmax)


def reject_extreme_values(data):
    # Reject extreme values
    return (data > -1e7) & (data < 1e7)


def reject_outliers(data, m=3):
    """
    Reject outliers beyond m standard deviations of the mean.
    :param data: numpy array containing data
    :param m: number of standard deviations from the mean. Default: 3
    """
    stdev = np.nanstd(data)
    if stdev > 0.0:
        ind = abs(data - np.nanmean(data)) < m * stdev
    else:
        ind = len(data) * [True]

    return ind

def return_array_subsites_standard_loc(array):
    DBurl= 'https://datareview.marine.rutgers.edu/regions/view/{}.json'.format(array)
    url_ct = get_url_content(DBurl)['region']['sites']
    loc_df = pd.DataFrame()
    for ii in range(len(url_ct)):
        if url_ct[ii]['reference_designator'] != 'CP05MOAS':
            data = {
                    'lat': url_ct[ii]['latitude'],
                    'lon': url_ct[ii]['longitude'],
                    'max_z': url_ct[ii]['max_depth']
                    }
            new_r = pd.DataFrame(data, columns=['lat', 'lon', 'max_z'], index=[url_ct[ii]['reference_designator']])
            loc_df = loc_df.append(new_r)
    return loc_df

def return_stream_vars(stream):
    # return all variables that should be found in a stream (from the data review database)
    stream_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        stream_vars.append(p['name'])
    return stream_vars


def return_raw_vars(ds_variables):
    # return a list of raw variables (eliminating engineering, qc, and timestamps)
    misc_vars = ['quality', 'string', 'timestamp', 'deployment', 'provenance', 'qc', 'time', 'mission', 'obs',
                 'serial_number', 'volt', 'ref', 'sig', 'amp', 'rph', 'calphase', 'phase', 'checksum', 'description',
                 'product_number']
    reg_ex = re.compile('|'.join(misc_vars))
    raw_vars = [s for s in ds_variables if not reg_ex.search(s)]
    return raw_vars


def return_science_vars(stream):
    # return only the science variables (defined in preload) for a data stream
    sci_vars = []
    dr = 'http://datareview.marine.rutgers.edu/streams/view/{}.json'.format(stream)
    r = requests.get(dr)
    params = r.json()['stream']['parameters']
    for p in params:
        if p['data_product_type'] == 'Science Data':
            sci_vars.append(p['name'])
    return sci_vars


def stream_word_check(method_stream_dict):
    # check stream names for cases where extra words are used in the names
    omit_word = ['_dcl', '_imodem', '_conc']
    mm = []
    ss = []
    ss_new = []

    for y in method_stream_dict.keys():
        mm.append(str(y).split('-')[0])
        ss.append(str(y).split('-')[1])

    for s in ss:
        wordi = []
        for word in omit_word:
            if word in s:
                wordi.append(word)
                break

        if wordi:
            fix = s.split(wordi[0])
            if len(fix) == 2:
                ss_new.append(fix[0] + fix[1].split('_recovered')[0])
        elif '_recovered' in s:
            ss_new.append(s.split('_recovered')[0])

        else:
            ss_new.append(s)
    return pd.DataFrame({'method': mm, 'stream_name': ss, 'stream_name_compare': ss_new})


def timestamp_gap_test(df):
    gap_list = []
    df['diff'] = df['time'].diff()
    index_gap = df['diff'][df['diff'] > pd.Timedelta(days=1)].index.tolist()
    for i in index_gap:
        gap_list.append([pd.to_datetime(str(df['time'][i-1])).strftime('%Y-%m-%dT%H:%M:%S'),
                         pd.to_datetime(str(df['time'][i])).strftime('%Y-%m-%dT%H:%M:%S')])
    return gap_list


def variable_statistics(var_data, stdev=None):
    """
    Calculate statistics for a variable of interest
    :param variable: array containing data
    :param stdev: desired standard deviation to exclude from analysis
    """
    if stdev is None:
        varD = var_data
        num_outliers = None
    else:
        ind = reject_extreme_values(var_data)
        var = var_data[ind]

        ind2 = reject_outliers(var, stdev)
        varD = var[ind2]
        varD = varD.astype('float64')  # force variables to be float64 (float32 is not JSON serializable)
        num_outliers = len(var_data) - len(varD)

    mean = round(np.nanmean(varD), 4)
    min = round(np.nanmin(varD), 4)
    max = round(np.nanmax(varD), 4)
    sd = round(np.nanstd(varD), 4)
    n = len(varD)

    return [num_outliers, mean, min, max, sd, n]

def format_dates(dd):
    fd = dt.datetime.strptime(dd.replace(',', ''), '%m/%d/%y %I:%M %p')
    fd2 = dt.datetime.strftime(fd, '%Y-%m-%dT%H:%M:%S')
    return fd2


def reject_timestamps_in_groups(groups, d_groups, n_std, tt, yy, zz, inpercentile):
    y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_exclude = [], [], [], [], [], [], [], []

    tm = 1
    for ii in range(len(groups)):
        nan_ind = d_groups[ii + tm].notnull()
        if not nan_ind.any():
            print('{} {} {}'.format('group', ii, 'is all NaNs'))
            tm += 2
            continue
        xtime = d_groups[ii + tm][nan_ind]
        ypres = d_groups[ii + tm + 1][nan_ind]
        nval = d_groups[ii + tm + 2][nan_ind]
        tm += 2

        l_arr.append(len(nval))  # count of data to filter out small groups
        y_avg.append(ypres.mean())
        n_avg.append(nval.mean())
        n_min.append(nval.min())
        n_max.append(nval.max())

        if n_std is not None:
            upper_l = nval.mean() + n_std * nval.std()
            lower_l = nval.mean() - n_std * nval.std()
            n0_std.append(upper_l)
            n1_std.append(lower_l)
            t_ind = np.where((nval < lower_l) | (nval > upper_l), True, False)
            # d_ind = np.where((nval > lower_l) & (nval < upper_l), True, False)
        else:
            upper_l = np.percentile(nval, 100 - inpercentile)
            lower_l = np.percentile(nval, inpercentile)
            n0_std.append(upper_l)
            n1_std.append(lower_l)
            t_ind = np.where((nval < lower_l) | (nval > upper_l), True, False)
            # d_ind = np.where((nval > lower_l) & (nval < upper_l), True, False)

        time_exclude = np.append(time_exclude, xtime[t_ind].values)
        # n_data = np.append(n_data, nval[d_ind].values)
        # n_time = np.append(n_time, xtime[d_ind].values)
        # n_pres = np.append(n_pres, ypres[d_ind].values)

    time_to_exclude = np.unique(time_exclude)
    t_ex, z_ex, y_ex = reject_suspect_data(tt, yy, zz, time_to_exclude)
    # if len(time_to_exclude) != 0:
    #     t_ex = tt
    #     y_ex = yy
    #     z_ex = zz
    #     for row in time_to_exclude:
    #         ntime = pd.to_datetime(row)
    #         ne = np.datetime64(ntime)
    #
    #         ind = np.where((t_ex != ne), True, False)
    #         if not ind.any():
    #             print('{} {}'.format(row, 'is not in data'))
    #             print(np.unique(ind))
    #         else:
    #             t_ex = t_ex[ind]
    #             z_ex = z_ex[ind]
    #             y_ex = y_ex[ind]

    return y_avg, n_avg, n_min, n_max, n0_std, n1_std, l_arr, time_to_exclude, t_ex, z_ex, y_ex


def reject_timestamps_portal(subsite, r, tt, yy, zz):

    dr = pd.read_csv('https://datareview.marine.rutgers.edu/notes/export')
    drn = dr.loc[dr.type == 'exclusion']

    if len(drn) != 0:
        subsite_node = '-'.join((subsite, r.split('-')[1]))
        drne = drn.loc[drn.reference_designator.isin([subsite, subsite_node, r])]
        if len(drne['reference_designator']) != 0:
            t_ex = tt
            y_ex = yy
            z_ex = zz
            for ij, row in drne.iterrows():
                sdate = format_dates(row.start_date)
                edate = format_dates(row.end_date)
                ts = np.datetime64(sdate)
                te = np.datetime64(edate)
                if t_ex.max() < ts:
                    continue
                elif t_ex.min() > te:
                    continue
                else:
                    ind = np.where((t_ex < ts) | (t_ex > te), True, False)
                    if len(ind) != 0:
                        t_ex = t_ex[ind]
                        z_ex = z_ex[ind]
                        y_ex = y_ex[ind]
                        print('excluding {} timestamps [{} - {}]'.format(len(ind), sdate, edate))

    return t_ex, z_ex, y_ex


def add_pressure_to_dictionary_of_sci_vars(ds):
    y_unit = []
    y_name = []
    if 'MOAS' in ds.subsite:
        if 'CTD' in ds.sensor:  # for glider CTDs, pressure is a coordinate
            pressure = 'sci_water_pressure_dbar'
            y = ds[pressure].values
            if ds[pressure].units not in y_unit:
                y_unit.append(ds[pressure].units)
            if ds[pressure].long_name not in y_name:
                y_name.append(ds[pressure].long_name)
        else:
            pressure = 'int_ctd_pressure'
            y = ds[pressure].values
            if ds[pressure].units not in y_unit:
                y_unit.append(ds[pressure].units)
            if ds[pressure].long_name not in y_name:
                y_name.append(ds[pressure].long_name)
    else:
        pressure = pf.pressure_var(ds, ds.data_vars.keys())
        y = ds[pressure].values

    if len(y[y != 0]) == 0 or sum(np.isnan(y)) == len(y) or len(y[y != ds[pressure]._FillValue]) == 0:
        print('Pressure Array of all zeros or NaNs or fill values - ... using pressure coordinate')
        pressure = [pressure for pressure in ds.coords.keys() if 'pressure' in ds.coords[pressure].name]
        y = ds.coords[pressure[0]].values

    try:
        ds[pressure].units
        if ds[pressure].units not in y_unit:
            y_unit.append(ds[pressure].units)
    except AttributeError:
        print('pressure attributes missing units')
        if 'pressure unit missing' not in y_unit:
            y_unit.append('pressure unit missing')

    try:
        ds[pressure].long_name
        if ds[pressure].long_name not in y_name:
            y_name.append(ds[pressure].long_name)
    except AttributeError:
        print('pressure attributes missing long_name')
        if 'pressure long name missing' not in y_name:
            y_name.append('pressure long name missing')

    return y, y_unit, y_name


def reject_erroneous_data(t, y, z, fill_value):

    '''

    :param t: time array
    :param y: pressure array
    :param z: data values
    :param fill_value: fill values defined in the data file
    :return: filtered data from fill values, NaNs, extreme values '|1e7|' and data outside global ranges
    '''

    # reject fill values
    fv_ind = z != fill_value
    y_nofv = y[fv_ind]
    t_nofv = t[fv_ind]
    z_nofv = z[fv_ind]
    print(len(z) - len(fv_ind), ' fill values')

    # reject NaNs
    nan_ind = ~np.isnan(z_nofv)
    t_nofv_nonan = t_nofv[nan_ind]
    y_nofv_nonan = y_nofv[nan_ind]
    z_nofv_nonan = z_nofv[nan_ind]
    print(len(z_nofv) - len(nan_ind), ' NaNs')

    # reject extreme values
    ev_ind = cf.reject_extreme_values(z_nofv_nonan)
    t_nofv_nonan_noev = t_nofv_nonan[ev_ind]
    y_nofv_nonan_noev = y_nofv_nonan[ev_ind]
    z_nofv_nonan_noev = z_nofv_nonan[ev_ind]
    print(len(z_nofv_nonan) - len(ev_ind), ' Extreme Values', '|1e7|')

    # reject values outside global ranges:
    global_min, global_max = cf.get_global_ranges(r, vinfo['var_name'])
    if isinstance(global_min, (int, float)) and isinstance(global_max, (int, float)):
        gr_ind = cf.reject_global_ranges(z_nofv_nonan_noev, global_min, global_max)
        t_nofv_nonan_noev_nogr = t_nofv_nonan_noev[gr_ind]
        y_nofv_nonan_noev_nogr = y_nofv_nonan_noev[gr_ind]
        z_nofv_nonan_noev_nogr = z_nofv_nonan_noev[gr_ind]
        print('{} Global ranges [{} - {}]'.format(len(z_nofv_nonan_noev) - len(gr_ind),
                                                  global_min, global_max))
    else:
        gr_ind = []
        t_nofv_nonan_noev_nogr = t_nofv_nonan_noev
        y_nofv_nonan_noev_nogr = y_nofv_nonan_noev
        z_nofv_nonan_noev_nogr = z_nofv_nonan_noev
        print('{} global ranges [{} - {}]'.format(len(gr_ind), global_min, global_max))

    return t_nofv_nonan_noev_nogr, y_nofv_nonan_noev_nogr, z_nofv_nonan_noev_nogr


def reject_suspect_data(t, y, z, timestamps):
    tt = t
    yy = y
    zz = z
    for row in timestamps:
        ntime = pd.to_datetime(row)
        ne = np.datetime64(ntime)
        ind = np.where((tt != ne), True, False)
        if not ind.any():
            print('{} {}'.format(row, 'is not in data'))
            print(np.unique(ind))
        else:
            tt = tt[ind]
            zz = zz[ind]
            yy = yy[ind]

    return tt, yy, zz
