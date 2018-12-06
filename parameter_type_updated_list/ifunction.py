
# check module

import requests
import pandas as pd


def read_file(file, col_num):
    file_content = pd.read_csv(file)
    col_content = list(file_content[file_content.columns[col_num]])
    return col_content


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
        url_info = r.json()
    return url_info


if __name__ == '__main__':
    main()