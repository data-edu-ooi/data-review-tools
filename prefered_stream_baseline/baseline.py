#! /Users/leila/miniconda2/bin/python

import requests
import pandas as pd

outdir = '/Users/leila/Documents/NSFEduSupport/test/'
methods_base = ['recovered_inst', 'recovered_host', 'recovered_cspp', 'recovered_wfp','telemetered']
base_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
not_4_review = ['ENG', 'MOPAK', 'HYD', 'OBSSP', 'OBSBB', 'CAM', 'FLOBN', 'RASFL', 'OSMOI', 'PPSDN', 'MASSP']
non_valid_method = ['bad']

def notin_list(x, ix):
    # filter out list entries with specific words.
    y = [el for el in x if not any(ignore in el for ignore in ix)]
    return y

def in_list(x, ix):
    y = [el for el in x if any(ignore in el for ignore in ix)]
    return y

def check_duplicates(a):
    dupes = [x for n, x in enumerate(a) if x in a[:n]]
    return dupes

def remove_duplicates(l):
    return list(set(l))


def check_length(x):
    y = []
    for item in x:
        if len(item) == 12:
            y.append(item)
    return y

def get_url_content(url_address):
    # get content of a url in a json format
    r = requests.get(url_address)
    if r.status_code is not 200:
        print r.reason
        print('Problem with ', url_address)
    else:
        url_content = r.json()
    return url_content


arrays = get_url_content(base_url)
print('arrays_count: ', len(arrays))


df_both = pd.DataFrame()
df_notin_database = pd.DataFrame()
count = 0
for ii in range(len(arrays)):
    print('working on: ', arrays[ii])
    url_a = '{:s}/{:s}'.format(base_url, arrays[ii])
    nodes = get_url_content(url_a) # grab the list of nodes
    for jj in range(len(nodes)):
        url_n = '{:s}/{:s}/{:s}'.format(base_url, arrays[ii],nodes[jj])
        all_sensor = get_url_content(url_n) # grab the list of sensors
        sample_sensors = notin_list(all_sensor, not_4_review)
        if len(sample_sensors) != 0:
            sensors = check_length(sample_sensors)
            if len(sensors) != 0:
                for nn in range(len(sensors)):
                    url_s = '{:s}/{:s}/{:s}/{:s}'.format(base_url, arrays[ii], nodes[jj],sensors[nn])
                    methods = get_url_content(url_s) # grab the list of methods
                    y = notin_list(methods, non_valid_method) # drop non valid methods
                    if len(y) != 1: # sort methods in order of preference
                        z = sorted(y, key=lambda zz: methods_base.index(zz)) # sorted method list
                    else:
                        z = y # one method exist

                    # get the stream list from database to filter out the engineering streams
                    refdes = '{:s}-{:s}-{:s}'.format(arrays[ii], nodes[jj], sensors[nn])
                    url_databse = 'https://datareview.marine.rutgers.edu/instruments/view/' + refdes + '.json'

                    if len(z) != 0:
                        url_uFrame = '{:s}/{:s}/{:s}/{:s}/{:s}'.format(base_url, arrays[ii], nodes[jj], sensors[nn],
                                                                       z[0])
                        streams = get_url_content(url_uFrame)  # grab uFrame list of streams

                        try:
                            print('working on: ', refdes)
                            stream_DB = get_url_content(url_databse)['instrument']['data_streams'] # grab database list of streams

                            n_stream = []
                            for mm in range(len(stream_DB)):
                                x = stream_DB[mm]['method']
                                if x == z[0]:
                                    xx = stream_DB[mm]['stream']['stream_type']
                                    if xx == 'Science':
                                        i_stream = stream_DB[mm]['stream']['name']
                                        n_stream.append(i_stream)

                            if len(n_stream) != 0:
                                stream_base = in_list(streams, n_stream)
                                status = check_duplicates(stream_base)
                                if len(status) != 0:
                                    text_comment = str(status) + 'duplicated in uFrame'
                                else:
                                    text_comment = ''

                                stream_base = remove_duplicates(stream_base)

                                for kk in range(len(stream_base)):
                                    text_db = ' yes'
                                    text_uframe = 'yes'
                                    id_both = pd.DataFrame({
                                                        "refdes": refdes,
                                                        "methods": str(z[0]),
                                                        "streams-uFrame": str(stream_base[kk]),
                                                        "database": text_db,
                                                        "uFrame": text_uframe,
                                                        "comment": text_comment
                                                        }, index=[count])
                                    df_both = df_both.append(id_both)
                                    count += 1
                            else:
                                print(arrays[ii], nodes[jj], sensors[nn], z)
                                print(n_stream,'no science streams in database')
                                text_db = 'yes'
                                text_uframe = 'yes'
                                text_comment = 'no science streams in database'
                                id_notin_database = pd.DataFrame({
                                                        "refdes": refdes,
                                                        "methods": str(z[0]),
                                                        "streams-uFrame": str(streams),
                                                        "database": text_db,
                                                        "uFrame": text_uframe,
                                                        "comment": text_comment
                                                        }, index=[count])
                                df_notin_database = df_notin_database.append(id_notin_database)
                                count += 1

                        except UnboundLocalError:
                            print(arrays[ii], nodes[jj], sensors[nn], 'no refdes in database')
                            text_db = ' no'
                            text_uframe = 'yes'
                            text_comment = 'no refdes in database'

                            id_notin_database = pd.DataFrame({
                                                "refdes": refdes,
                                                "methods": str(z[0]),
                                                "streams-uFrame": str(streams),
                                                "database": text_db,
                                                "uFrame": text_uframe,
                                                "comment": text_comment
                                                }, index=[count])
                            df_notin_database = df_notin_database.append(id_notin_database)
                            count += 1
                    else:
                        try:
                            print(arrays[ii], nodes[jj], sensors[nn], 'method not valid', methods)
                            stream_DB = get_url_content(url_databse)['instrument']['data_streams']
                            text_db = ' yes'
                            text_uframe = 'yes'
                            text_comment = 'method not valid'
                            id_notin_database = pd.DataFrame({
                                                    "refdes": refdes,
                                                    "methods": str(methods),
                                                    "streams-uFrame": str(z),
                                                    "database": text_db,
                                                    "uFrame": text_uframe,
                                                    "comment": text_comment
                                                    }, index=[count])
                            df_notin_database = df_notin_database.append(id_notin_database)
                            count += 1
                        except:
                            print(arrays[ii], nodes[jj], sensors[nn], 'no refdes in DB',
                                  'method not valid', z)

                            text_db = ' no'
                            text_uframe = 'yes'
                            text_comment = 'method not valid'

                            id_notin_database = pd.DataFrame({
                                                "refdes": refdes,
                                                "methods": str(methods),
                                                 "streams-uFrame": str(z),
                                                "database": text_db,
                                                "uFrame": text_uframe,
                                                "comment": text_comment
                                                }, index=[count])
                            df_notin_database = df_notin_database.append(id_notin_database)
                            count += 1


header = ['refdes', 'methods', 'streams-uFrame','database','uFrame','comment']
df_both.to_csv(outdir + 'stream_review_baseline.csv', columns= header, index=False)
df_notin_database.to_csv(outdir + 'stream_notin_databse.csv', columns=header, index=False)



