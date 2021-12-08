import sys
import numpy as np
import pandas as pd
import aif360
from aif360 import datasets
# from aif360.algorithms.preprocessing import DisparateImpactRemover

# read in data and convert to AIF format
df = pd.read_csv('data/COMPAS_processed.csv')
binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=df,
    label_names=['two_year_recid'],
    protected_attribute_names=['sex', 'race'])
print(binaryLabelDataset)

# make training splits

# compute fairness on original data

# mitigate bias on dataset

# compute fairness on mitigated dataset
