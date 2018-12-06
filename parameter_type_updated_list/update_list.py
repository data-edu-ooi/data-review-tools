#! /Users/leila/miniconda2/bin/python

import pandas as pd

file = 'parameter_list_from_database_manual_update.csv'
file_2_update = 'add_parameter_type.csv'

drop_columns = {'region','sensor','site','method','stream_name','stream_content','stream_description',
                'parameter_fill_value', 'parameter_data_product_identifier', 'parameter_type',
                'parameter_precision', 'parameter_value_encoding', 'parameter_function_id',
                'parameter_function_map', 'GlobalRangeMin', 'GlobalRangeMax', '_DataLevel', '_Units',
                'parameter_unit', 'parameter_standard_name', 'parameter_display_name',
                'parameter_data_level', 'parameter_data_product_type', 'parameter_description'
                }

ed = pd.read_csv(file_2_update)
list_item = ed['parameter_id'].values.tolist()

qc = pd.read_csv(file)

nf = qc[(qc['add_parameter_type'] == 'Science Data') & (qc['Type'].isnull())]
nf = nf.drop(columns=drop_columns)
nf = nf.drop_duplicates()


ind = (nf[nf['parameter_id'].isin(list_item)].index).tolist()
if len(ind) == 0:
    with open('add_parameter_type.csv', 'a') as f:
        f.write('\n')
        nf.to_csv(f, header=False, index=False)
else:
    mf = nf.drop(index=ind)
    with open('add_parameter_type.csv', 'a') as f:
        f.write('\n')
        mf.to_csv(f, header=False, index=False)

#nf.to_csv( 'updated_list.csv', index=False)
