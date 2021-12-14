import numpy as np
import pandas as pd
from autoimpute.imputations import SingleImputer, MultipleImputer, MiceImputer

imp = MiceImputer(
    n=10,
    strategy={"sex": "pmm", "priors_count": "pmm"},
    predictors={"sex": "all", "priors_count": "all"},
    imp_kwgs={"pmm": {"fill_value": "random"}},
    visit="left-to-right",
    return_list=True
)
data = pd.read_csv('COMPAS_processed_missing.csv')

data_imputed = imp.fit_transform(data)
data_imputed.to_csv('imputted_pmm.csv')
