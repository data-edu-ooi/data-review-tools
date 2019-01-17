# check module

import pandas as pd
import numpy as np
from pandas import Series
from pandas import Grouper
from pandas import DataFrame
from pandas import concat


def group_by_timerange(data_x, data_y, g_freq):

    series = pd.DataFrame(columns=['Date', 'DO'], index=data_x)
    series['Date'] = data_x
    series['DO'] = data_y
    groups = series.groupby(Grouper(freq=g_freq))

    g_data = concat([DataFrame(x[1].values) for x in groups], axis=1)
    g_data = DataFrame(g_data)
    g_data.columns = range(1, len(g_data.columns) + 1)

    return groups, g_data

