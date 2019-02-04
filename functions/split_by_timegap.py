# check module

import numpy as np


def split_by_timegap(data, gap_in_seconds):
    data['time'] = data.index
    data['gap'] = (data['time'].diff()).dt.seconds > gap_in_seconds
    ind_TF = np.nonzero(data['gap'][1:].values != data['gap'][:-1].values)[0] + 1
    if ind_TF.size > 0:
        indices = []
        for ii in range(len(ind_TF)):
            if not data['gap'][ind_TF[ii]]:
                indices.append(ind_TF[ii])

        b = np.split(data, ind_TF)
        b = b[0::2] if 'False' else b[1::2]
        print(indices)
        print(b)
        return b
