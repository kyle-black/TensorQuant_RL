import pandas as pd


df =pd.read_csv('eurusd_20000_2.csv')

df.drop_duplicates(inplace=True)

df.to_csv('eurusd_20000_3.csv')