import pandas as pd
import random

# read data
data = pd.read_csv("compas-scores-two-years.csv", header=0, usecols=['id', 'sex', 'age_cat', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid'])

# remove other races
data = data.drop(data[((data.race != 'African-American') & (data.race != 'Caucasian'))].index)

# make race 1 hot
data['race'] = (data['race'] == 'African-American').astype(int)

# make sex 1 hot
data['sex'] = (data['sex'] == 'Female').astype(int)

# make charge 1 hot
data['c_charge_degree'] = (data['c_charge_degree'] == 'F').astype(int)

# make age categories numerical (0, 1, 2 alphabetically)
data['age_cat'] = pd.Categorical(data['age_cat'])
data['age_cat'] = data['age_cat'].cat.codes

# equalize number of white / black entries
print(data['race'].value_counts())

data = data.drop(data[data['race'] == 1].sample(frac=0.336).index)

print(data)
data.to_csv("COMPAS_processed.csv")

# randomly create missing values
data.loc[data.sample(frac=0.08).index, 'priors_count'] = ""
data.loc[data.sample(frac=0.08).index, 'sex'] = ""

print(data.info())
data.to_csv("COMPAS_processed_missing.csv")
