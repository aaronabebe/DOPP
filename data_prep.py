import pandas as pd 

# read source csv
raw = pd.read_csv("unesco_poverty_dataset.csv") 
keys = raw.DEMO_IND.unique() 

# define base csv
base = raw[['LOCATION', 'TIME']]

# for every var join on LOCATION & TIME 
for i in range(0,len(keys)):
    loop = raw.loc[raw.DEMO_IND == keys[i]]
    base = pd.merge(base, loop[['LOCATION', 'TIME', 'Value']],  how='left', left_on=['LOCATION','TIME'], right_on = ['LOCATION','TIME']) 
    base.columns = base.columns.str.replace('Value', keys[i])

# export
base.to_csv('transformed.csv', sep=',', na_rep="NA")
exit