import pandas as pd

data = pd.read_csv('COMPAS_processed_missing.csv')
data = data.dropna()
print(data)
data.to_csv('row_deletion.csv')
