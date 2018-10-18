#!/usr/bin/env python
"""
Created on 10/17/2018
@author Lori Garzio
@brief Define the delivery method and data stream(s) preferred for analysis
"""


import os
import numpy as np
import json


def main(f):
    sdict = dict()
    data = json.load(open(f, 'r'))
    refdes = data['refdes']
    deployments = np.sort(data['deployments'].keys()).tolist()
    method_list = ['streamed', 'recovered_inst', 'recovered_wfp', 'recovered_cspp', 'recovered_host', 'telemetered']
    for d in deployments:
        info = []
        ddata = data['deployments'][d]
        for m in ddata['method'].keys():
            for s in ddata['method'][m]['stream'].keys():
                for fname in ddata['method'][m]['stream'][s]['file'].keys():
                    fsummary = ddata['method'][m]['stream'][s]['file'][fname]
                    nt = fsummary['n_timestamps']
                    info.append([str(m), str(s), nt])

        for meth in method_list:
            x = [k for k in info if meth in k[0]]
            if len(x) > 0:
                break

        preferred_method_stream = []
        for i in x:
            preferred_method_stream.append('-'.join((i[0], i[1])))

        if d not in sdict.keys():
            sdict.update({d: preferred_method_stream})
    print sdict

    sfile = '{}/{}-preferred_stream.json'.format(os.path.dirname(f), refdes)
    with open(sfile, 'w') as outfile:
        json.dump(sdict, outfile)


if __name__ == '__main__':
    f = '/Users/lgarzio/Documents/repo/OOI/data-edu-ooi/data-review-tools/data_review/output/GP03FLMA/GP03FLMA-RIM01-02-CTDMOG040/GP03FLMA-RIM01-02-CTDMOG040-file_analysis.json'
    main(f)
