import pandas as pd

df = pd.read_excel('data_in_for_Inf.xlsx')
df.to_csv('data_in_for_Inf.csv', index=False)
