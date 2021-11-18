import pandas as pd
import random

# read data
data = pd.read_csv("data/compas-scores-two-years.csv", header=0, usecols=['sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid'])

# remove other races
data = data.drop(data[((data.race != 'African-American') & (data.race != 'Caucasian'))].index)

# make race 1 hot
data['race'] = (data['race'] == 'African-American').astype(int)

# equalize number of white / black entries
print(data['race'].value_counts())

data = data.drop(data[data['race'] == 1].sample(frac=0.336).index)

print(data['race'].value_counts())

# randomly create missing values
data.loc[data.sample(frac=0.08).index, 'priors_count'] = ""
data.loc[data.sample(frac=0.08).index, 'sex'] = ""

print(data['sex'].value_counts())

print(data)
data.to_csv("data/COMPAS_processed.csv")
