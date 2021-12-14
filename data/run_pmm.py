import numpy as np
import pandas as pd
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer

imp = MultipleImputer()
data = pd.read_csv('COMPAS_processed_missing.csv')

data_imputed = imp.fit_transform(data)
data_imputed.to_csv('imputted_pmm.csv')
