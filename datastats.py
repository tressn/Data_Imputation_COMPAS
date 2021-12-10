import pandas as pd

# data = pd.read_csv('data/COMPAS_processed.csv')
missing_data = pd.read_csv('data/COMPAS_processed_missing.csv')
# orig_data = pd.read_csv('data/compas-scores-two-years.csv')
print(missing_data)

print(missing_data['race'].value_counts())
print(missing_data['sex'].value_counts())
print(missing_data['priors_count'].value_counts())

print(missing_data.isnull().sum().sum())
print(missing_data['sex'].isnull().sum().sum())
print(missing_data['priors_count'].isnull().sum().sum())
