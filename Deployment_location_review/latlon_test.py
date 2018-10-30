#! /usr/local/bin/python

"""
Created on Wednesday Oct 24 2018
@author: leila

The script purpose are twofold:
(1) check the deployment sheets in asset management if lat and lon information per deployments are unique or different.
(2) calculate the difference in km between deployments location for a platform or an instrument.

The script's input items:
- asset management deployment sheets
- database list of platforms

"""

import pandas as pd
import requests
import urllib2
from geopy.distance import geodesic

pd.set_option('mode.chained_assignment', None)

output_dir = '/Users/leila/Documents/NSFEduSupport/github/data-review-tools/Deployment_location_review'
file_4_diagnosis = output_dir + '/diagnosis/'
file_4_difference = output_dir + '/difference/'
file_4_unique = output_dir + '/unique/'

asset_deployment = 'https://raw.githubusercontent.com/ooi-integration/asset-management/master/deployment/'
item_database = 'https://datareview.marine.rutgers.edu/'
cabled_list = ['CE02SHBP', 'CE04OSBP', 'CE04OSPD', 'CE04OSPS','RS']

def get_url_content(url_address):
    # get content of a url in a json format
    r = requests.get(url_address)
    if r.status_code is not 200:
        print r.reason
        print('Problem wi chatth', url_address)
    else:
        url_content = r.json()
    return url_content

regions = []
r_list = get_url_content(item_database + 'regions.json')['regions']
for ii in range(len(r_list)):
    regions.append(r_list[ii]['reference_designator'])

platforms = []
for region in regions:
    p_list = get_url_content(item_database + 'regions/view/' + region + '.json')['region']['sites']
    for ii in range(len(p_list)):
        name = p_list[ii]['reference_designator']
        if '05MOAS' in str(name):
            nodes = get_url_content(item_database + 'sites/view/' + str(name) + '.json')['site']['nodes']
            for mm in range(len(nodes)):
                platform = nodes[mm]['reference_designator']
                platforms.append(platform)
        else:
            platform = str(name)
            platforms.append(platform)


def list_url(platforms):
    # create a url and list the broken ones and to what platforms they correspond.
    urls = []
    nf = pd.DataFrame()
    omit_HTTPError = []
    count = 0
    for platform in platforms:
        url = asset_deployment + platform + '_Deploy.csv'
        urls.append(url)
        try:
            asset_data = pd.read_csv(url)
        except urllib2.HTTPError:
            print('No info in: ', url)
            omit_HTTPError.append((url.split('/')[-1]).split('_Deploy.csv')[0])
            nfi = pd.DataFrame({'url': [url], 'message': ['HTTPError']}, index=[count])
            count += 1
            nf = nf.append(nfi)
    return (nf, omit_HTTPError, urls)


def filter_url(url_list, words):
    # filter out urls with specific words.
    for word in words:
        url_filtered_list = [url for url in url_list if word not in url]
        url_list = url_filtered_list
    return url_filtered_list


def sample_url(url_list,words):
    # select urls with specific words.
    url_subset_list = []
    for word in words:
        subset_list = [url for url in url_list if word in url]
        url_subset_list += subset_list
    return url_subset_list

def loc_difference(location, name, difference):
    # calculate the difference in km between deployments location for a platform or an instrument.
    y = {}
    icount = 0
    for i, k in location.iterrows():
        if i > 0 and len(location) > 1:
            loc1 = [k['lat'], k['lon']]
            d1 = int(k['deploymentNumber'])
            for x in range(i):
                info0 = location.iloc[x]
                compare = 'diff_km_D{}_to_D{}'.format(d1, int(info0['deploymentNumber']))
                loc0 = [info0['lat'], info0['lon']]
                diff_loc = round(geodesic(loc0, loc1).kilometers, 4)
                y[name + '_' + str(icount)] = {}
                y[name + '_' + str(icount)].update(
                    {'name': compare, 'difference': diff_loc, 'platform': name})
                icount += 1

    dn = pd.DataFrame.from_dict(y, orient='index').sort_index()
    difference = difference.append(dn)
    return difference


def platform_location_list(asset_data, count, platform_name, nudf, udf, dl):
    # for each platform:
    # check if the location (lat and lon) for each deployment is unique or different and report out the list.
    # compile a list of deployments with unique location and calculate the difference in km between deployments.

    column_list = list(pd.unique(asset_data['deploymentNumber'].ravel()))
    deploy_loc = {}
    ncount = 0
    for num in column_list:
        ind_s = asset_data.loc[(asset_data['deploymentNumber'] == num)]

        # check uniqueness of lat and lon:
        lat_s = len(list(pd.unique(ind_s['lat'].ravel())))
        lon_s = len(list(pd.unique(ind_s['lon'].ravel())))
        if lat_s > 1 or lon_s > 1:
            # print(num, ': unique test - Fail')
            idf = pd.DataFrame({'total deployments': len(column_list), 'deployment': num, 'platform': [platform_name],
                                'message': ['Not Unique']}, index=[count])
            count += 1
            nudf = nudf.append(idf)
        else:
            # print(num, ': unique test - Pass')
            sdf = pd.DataFrame({'total deployments': len(column_list), 'deployment': num, 'platform': [platform_name],
                                'message': ['Unique']}, index=[count])
            udf = udf.append(sdf)
            count += 1

            deploy_loc[ncount] = {}
            nn = ind_s.index[0]
            deploy_loc[ncount]['platform'] = platform_name
            deploy_loc[ncount]['deploymentNumber'] = ind_s['deploymentNumber'][nn]
            deploy_loc[ncount]['lat'] = ind_s['lat'][nn]
            deploy_loc[ncount]['lon'] = ind_s['lon'][nn]
            ncount += 1

    # put info in a data frame
    df = pd.DataFrame.from_dict(deploy_loc, orient='index').sort_index()
    dl = loc_difference(df, platform_name, dl)
    return (df, udf, nudf, dl)

def instrument_location_list(asset_data, dl):
    # for each instrument:
    # compile a list of deployments and calculate the difference in km between deployments.
    column_list = list(pd.unique(asset_data['Reference Designator'].ravel()))
    for num in column_list:
        ind_s = asset_data.loc[(asset_data['Reference Designator'] == num)]
        ind_s = ind_s.reset_index(drop=True)
        dl = loc_difference(ind_s, num, dl)
    return dl

def location_check(url_4_loc,text_filter):
    # for each deployment sheet
    # drop rows with # character
    #
    dl = pd.DataFrame()
    nudf = pd.DataFrame()
    udf = pd.DataFrame()
    count = 0
    for url in url_4_loc:
        print(url)
        platform_name = (url.split('/')[-1]).split('_Deploy.csv')[0]
        asset_data = pd.read_csv(url)
        asset_data = asset_data[asset_data['CUID_Deploy'].str.contains('#') == False]
        if text_filter is 'any':
            (df, udf, nudf, dl) = platform_location_list(asset_data, count, platform_name, nudf, udf, dl)
        else:
            dl = instrument_location_list(asset_data, dl)

    return (dl, udf, nudf)


(nf, omit_HTTPError, urls) = list_url(platforms)
DF_HTTPErorr_header = ['url', 'message']
nf.to_csv(file_4_diagnosis + 'database_HTTPError.csv', columns = DF_HTTPErorr_header, index = True)

# Uncabled
url_filtered_list = filter_url(urls,cabled_list)
(df_uncabled, df_unique, df_non_unique)= location_check(url_filtered_list, 'any')
if df_unique.empty is False and df_non_unique.empty is False:
    df_uncabled_diagnosis = pd.merge(df_unique, df_non_unique, on=['platform', 'message'], how='outer')

    df_uncabled_diagnosis = df_uncabled_diagnosis.sort_values('platform')
    df_uncabled_diagnosis.to_csv(file_4_diagnosis + 'uncabled_loc-diagnosis_diff.csv', index=False)
else:
    if df_unique.empty is False:
        list_unique_platform = df_unique.drop_duplicates('platform')
        list_unique_platform.to_csv(file_4_unique + 'uncabled_loc_unique.csv', columns = ['platform'], index=False)
    if df_non_unique.empty is False:
        list_nonunique_platform = df_non_unique.drop_duplicates('platform')
        list_nonunique_platform.to_csv(file_4_unique + 'uncabled_loc_not_unique.csv', columns = ['platform'], index=False)

df_uncabled.to_csv(file_4_difference + 'uncabled_platform_location_diff.csv', index=False)

# Cabled
url_subset_list = sample_url(urls,cabled_list)
url_filtered_list = filter_url(url_subset_list, omit_HTTPError)

(df_cabled, df_unique, df_non_unique) = location_check(url_filtered_list, 'any')
if df_unique.empty is False and df_non_unique.empty is False:
    df_cabled_diagnosis = pd.merge(df_unique, df_non_unique, on=['platform', 'message','deployment', 'total deployments'], how='outer')
    platform_list = list(pd.unique(df_cabled_diagnosis['platform'].ravel()))
    pcount = 0
    pnudf = pd.DataFrame()
    pudf = pd.DataFrame()
    for iplatform in platform_list:
        ind_platform = df_cabled_diagnosis.loc[(df_cabled_diagnosis['platform'] == iplatform)]
        if 'Not Unique' in list(ind_platform['message']):
            pnudf = pnudf.append(pd.DataFrame({'platform': iplatform}, index=[pcount]))
        else:
            pudf = pudf.append(pd.DataFrame({'platform': iplatform}, index=[pcount]))
        pcount += 1

    df_cabled_diagnosis = df_cabled_diagnosis.sort_values('platform')
    pnudf.to_csv(file_4_unique + 'cabled_loc_not_unique.csv', index=False)
    pudf.to_csv(file_4_unique + 'cabled_loc_unique.csv', index=False)
else:
    if df_unique.empty is False:
        list_unique_platform = df_unique.drop_duplicates('platform')
        list_unique_platform.to_csv(file_4_unique + 'cabled_loc_unique.csv', columns = ['platform'], index=False)
    if df_non_unique.empty is False:
        list_nonunique_platform = df_non_unique.drop_duplicates('platform')
        list_nonunique_platform.to_csv(file_4_unique + 'cabled_loc_not_unique.csv', index=False)

df_cabled = df_cabled[~df_cabled['platform'].isin(list(pnudf['platform']))]
df_cabled.to_csv(file_4_difference + 'cabled_platform_Location_diff.csv', columns = ['platform'], index=False)


url_nonunique_url = sample_url(url_filtered_list, list(pd.unique(df_non_unique['platform'])))
(df_cabled_instrument, df_y_unique, df_n_unique) = location_check(url_nonunique_url, 'none')
df_cabled_instrument.to_csv(file_4_difference + 'cabled_instrument_location_diff.csv', index=False)