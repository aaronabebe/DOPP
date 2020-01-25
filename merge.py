import pandas as pd 


base = pd.read_csv('transformed.csv', index_col=0)
continents = pd.read_csv('continents.csv', index_col=4)

print(base)
print(continents)

merged = pd.merge(base, continents, left_on=base['LOCATION'], right_index=True)
merged.to_csv('merged.csv', index=False)
print(merged)