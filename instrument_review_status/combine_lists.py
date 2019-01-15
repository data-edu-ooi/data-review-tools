
import pandas as pd



# Step 1 Users/leila/Documents/NSFEduSupport/database/
file_1 = 'not_in_preferred_stream_baseline.csv'

file_2 = 'https://raw.githubusercontent.com/ooi-data-lab/data-review-tools/master/prefered_stream_baseline/stream_review_baseline.csv'

f1 = pd.read_csv(file_1)
f1 = f1.rename(columns={'reference_designator': 'refdes'})
f1 = f1.drop(columns={'inst'})


f2 = pd.read_csv(file_2)
f2 = f2.drop(columns={'methods','streams-uFrame','comment'})
f2 = f2.drop_duplicates()

f12 = pd.merge(f2, f1, on=['refdes'], how='outer')

f12.to_csv('combined_lists.csv', index=False)

# Step 2
# created after 1 and 2 were combined and then manually edited
file_3 = 'instrument_review_status.csv'
f3 =  pd.read_csv(file_3)

f123 = pd.merge(f12,f3, on=['refdes'], how='outer')

f123.to_csv('to_verify.csv', index=False)