#! /Users/leila/miniconda2/bin/python
# -*- coding: utf-8 -*-

import codecs
import pandas as pd
import ifunction


base_url = 'https://datareview.marine.rutgers.edu/'
qctable = 'https://raw.githubusercontent.com/ooi-integration/qc-lookup/master/data_qc_global_range_values.csv'
maindir = '/Users/leila/Documents/NSFEduSupport/test/parameter/'
filedir = maindir + 'file_2_use/'
reviewed = filedir + 'GolbalRange_review-Sheet39.csv'
outdir = codecs.open(maindir + 'parameter_list_from_database.csv', 'w','utf-8')


regions = ifunction.read_file(filedir + 'regions.csv',0)
inst_not_4_review = ifunction.read_file(filedir + 'inst_not_4_review.csv',0)
plt_not_4_review = ifunction.read_file(filedir + 'plt_not_4_review.csv',0)
non_valid_method = ifunction.read_file(filedir + 'non_valid_method.csv',0)
var_not_4_review = ifunction.read_file(filedir + 'var_not_4_review.csv',0)
unit_not_4_review = ifunction.read_file(filedir + 'unit_not_4_review.csv',0)
df_header = ifunction.read_file(filedir + 'output_header.csv',0)

# read qc table
qc = pd.read_csv(qctable)
qc = pd.DataFrame(qc, columns= ['ParameterID_T', 'GlobalRangeMin', 'GlobalRangeMax', '_DataLevel', '_Units'])
qc = qc.drop_duplicates()
qc = qc.rename(columns={'ParameterID_T': 'parameter_name'})

# read my file
rq = pd.read_csv(reviewed)
rq = rq.drop_duplicates()
rq = rq.rename(columns={'ParameterID_T': 'parameter_name'})

count = 0
df = pd.DataFrame()

for region in regions:
    arrays = ifunction.get_url_content(base_url + 'regions/view/' + region + '.json')
    print(region, len(arrays['region']['sites']))
    for ii in range(len(arrays['region']['sites'])):
        site = arrays['region']['sites'][ii]['reference_designator']
        site = ifunction.notin_list([site], plt_not_4_review)
        if site:
            instruments = ifunction.get_url_content(base_url + 'sites/view/' + str(site[0]) + '.json')
            print(site, len(instruments['site']['nodes']))
            for jj in range(len(instruments['site']['nodes'])):
                if len(instruments['site']['nodes'][jj]['instruments']) != 0:
                    for nn in range(len(instruments['site']['nodes'][jj]['instruments'])):
                        instrument = instruments['site']['nodes'][jj]['instruments'][nn]['reference_designator']
                        sensor = ifunction.notin_list([instrument], inst_not_4_review)
                        if sensor:
                            streams = ifunction.get_url_content(base_url + 'instruments/view/' + str(sensor[0]) + '.json')
                            for mm in range(len(streams['instrument']['data_streams'])):
                                stream_type = streams['instrument']['data_streams'][mm]['stream']['stream_type']
                                if stream_type == 'Science':
                                    stream_method = streams['instrument']['data_streams'][mm]['method']
                                    stream_method = ifunction.notin_list([stream_method], non_valid_method)
                                    if stream_method:
                                        stream_name = streams['instrument']['data_streams'][mm]['stream']['name']
                                        stream_description = streams['instrument']['data_streams'][mm]['stream']['description']
                                        stream_content = streams['instrument']['data_streams'][mm]['stream']['stream_content']
                                        for ss in range(len(streams['instrument']['data_streams'][mm]['stream']['parameters'])):
                                            parameter_name  = streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]['name']
                                            name_sel = ifunction.notin_list([parameter_name], var_not_4_review)
                                            if name_sel:
                                                parameter_unit = streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["unit"]
                                                unit_sel = ifunction.notin_list([parameter_unit], unit_not_4_review)
                                                if unit_sel:
                                                    id = pd.DataFrame({
                                                        "region": str(region),
                                                        "site": str(site[0]),
                                                        "sensor": str(sensor[0]),
                                                        "method" : str(streams['instrument']['data_streams'][mm]['method']),
                                                        "stream_name" : str(streams['instrument']['data_streams'][mm]['stream']['name']),
                                                        "stream_description" : str(streams['instrument']['data_streams'][mm]['stream']['description']),
                                                        "stream_content" : str(streams['instrument']['data_streams'][mm]['stream']['stream_content']),
                                                        "parameter_id": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["id"]),
                                                        "parameter_name" : str(name_sel[0]),
                                                        "parameter_unit": str(unit_sel[0].encode('utf_8')),
                                                        "parameter_data_level": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["data_level"]),
                                                        "parameter_data_product_type": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["data_product_type"]),
                                                        "parameter_description": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["description"]),
                                                        "parameter_display_name": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["display_name"]),
                                                        "parameter_standard_name": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["standard_name"]),
                                                        "parameter_fill_value": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["fill_value"]),
                                                        "parameter_precision": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["parameter_precision"]),
                                                        "parameter_function_id": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss]["parameter_function_id"]),
                                                        "parameter_function_map": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss][ "parameter_function_map"]),
                                                        "parameter_data_product_identifier": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss][ "data_product_identifier"]),
                                                        "parameter_type": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss][ "parameter_type"]),
                                                        "parameter_value_encoding": str(streams['instrument']['data_streams'][mm]['stream']['parameters'][ss][ "value_encoding"])},
                                                        index=[count], columns=df_header)

                                                    df = df.append(id)
                                                    count += 1

df = df.drop_duplicates()
df = df.drop(df[df['parameter_value_encoding'] == 'string'].index)
df = df.drop(df[df['parameter_type'] == 'boolean'].index)
df = df.drop(df[df['parameter_type'] == 'category<int8:str>'].index)

# merge with qc table
kq = pd.merge(df, qc, on=['parameter_name'], how='outer')
# df_header = df_header + ['GlobalRangeMin', 'GlobalRangeMax', '_DataLevel', '_Units']

# merge with my file
kr = pd.merge(kq, rq, on=['parameter_name'], how='outer')
kr = kr.drop(kr[kr['stream_name'].isnull()].index)
kr = kr.drop(kr[kr['Type'] == 'remove'].index)

kr = kr.assign(add_parameter_type="")
kr.loc[((kr['parameter_data_product_type'] != 'Science Data') & ~(kr['Type'].isnull())), 'add_parameter_type'] = 'Science Data'

df_header = df_header + ['GlobalRangeMin', 'GlobalRangeMax', '_DataLevel', '_Units', 'Type', 'add_parameter_type']

kr.to_csv(maindir + 'parameter_list_from_database.csv', header= df_header, index=False)


# get unique parameter id
nf = kr.drop(kr[kr['add_parameter_type'] != 'Science Data'].index)
drop_columns = {'region','sensor','site','method','stream_name','stream_content','stream_description',
                'parameter_fill_value', 'parameter_data_product_identifier', 'parameter_type',
                'parameter_precision', 'parameter_value_encoding', 'parameter_function_id',
                'parameter_function_map', 'GlobalRangeMin', 'GlobalRangeMax', '_DataLevel', '_Units',
                'parameter_unit', 'parameter_standard_name', 'parameter_display_name',
                'parameter_data_level', 'parameter_data_product_type', 'parameter_description'
                }
nf = nf.drop(columns=drop_columns)
nf = nf.drop_duplicates()

nf.to_csv(maindir + 'add_parameter_type.csv', index=False)





