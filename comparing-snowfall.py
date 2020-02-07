#Author: Shivam Solanki
#!/usr/bin/env python

import pandas as pd

## read in the data
df = pd.read_csv("../data/snowfall.csv")

## subset the data to only the states of interest
mask = [True if s in ['CO','UT','VT'] else False for s in df['state'].values]
df1 = df[mask]

## create a pivot that looks at the specific data we are interested in
pivot = df1.groupby(['state'])['snowfall'].describe()
df1_pivot = pd.DataFrame({'count': pivot['count'],
                          'avg_snowfall': pivot['mean'],
                          'max_snowfall': pivot['max']})
print(df1_pivot)
