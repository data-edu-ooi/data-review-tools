#!/usr/bin/env python
"""
Created on 9/13/2019
@author Lori Garzio
@brief Define the delivery method and data stream(s) preferred for analysis
"""


import os
import json
import glob


def main(directory):
    ff = glob.glob(directory + 'deployment*file_analysis.json')
    deployments = [x.split('/')[-1].split('-')[0] for x in ff]
    deployments.sort()
    sdict = dict()
    method_list = ['streamed', 'recovered_inst', 'recovered_wfp', 'recovered_cspp', 'recovered_host', 'telemetered']
    for d in deployments:
        f = [y for y in ff if d in y]
        data = json.load(open(f[0], 'r'))
        try:
            refdes = data['refdes']
        except KeyError:
            continue
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

    sfile = '{}/{}-preferred_stream.json'.format(os.path.dirname(f[0]), refdes)
    with open(sfile, 'w') as outfile:
        json.dump(sdict, outfile)

    return sdict


if __name__ == '__main__':
    directory = '/Users/lgarzio/Documents/repo/OOI/ooi-data-lab/data-review-tools/data_review/output/CE02SHBP/CE02SHBP-LJ01D-08-OPTAAD106/'
    main(directory)
