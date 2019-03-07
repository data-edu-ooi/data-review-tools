# check module

import pandas as pd
import numpy as np
from pandas import Series
from pandas import Grouper
from pandas import DataFrame
from pandas import concat


def group_by_time_range(data_x, data_y, g_freq):

    series = pd.DataFrame(columns=['Date', 'DO'], index=data_x)
    series['Date'] = data_x
    series['DO'] = data_y
    groups = series.groupby(Grouper(freq=g_freq))

    g_data = concat([DataFrame(x[1].values) for x in groups], axis=1)
    g_data = DataFrame(g_data)
    g_data.columns = range(1, len(g_data.columns) + 1)

    return groups, g_data

def group_by_time_frequency(data_x, data_y, columns, g_freq):

    data_in = pd.DataFrame(columns=columns, index=data_x)
    data_in['time'] = data_x
    data_in[columns[1:len(columns)]] = data_y
    groups = data_in.groupby(Grouper(freq=g_freq))

    d_groups = concat([DataFrame(x[1].values) for x in groups], axis=1)
    d_groups = DataFrame(d_groups)
    d_groups.columns = range(1, len(d_groups.columns) + 1)

    return groups, d_groups

def group_by_depth_range(t, z, x, columns, ranges):

    data_in = pd.DataFrame(columns=columns, index=t)
    data_in[columns[0]] = t
    data_in[columns[1]] = z
    data_in[columns[2]] = x

    groups = data_in.groupby(pd.cut(data_in[columns[1]], ranges))

    d_groups = concat([DataFrame(x[1].values) for x in groups], axis=1)
    d_groups = DataFrame(d_groups)
    d_groups.columns = range(1, len(d_groups.columns) + 1)

    return groups, d_groups
