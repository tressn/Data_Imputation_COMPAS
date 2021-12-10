import sys
import numpy as np
import pandas as pd
import aif360
from aif360 import datasets
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from collections import defaultdict

def test(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['accuracy'].append(metric.accuracy())
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['err_rate_diff'].append(metric.error_rate_difference())

    return metric_arrs


# setup
sys.path.insert(1, "../")
np.random.seed(0)

# read in data and convert to AIF format
df = pd.read_csv('data/COMPAS_processed.csv')
binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=df,
    label_names=['two_year_recid'],
    protected_attribute_names=['sex', 'race'])

privileged_groups_race = [{'race': 0}]
unprivileged_groups_race = [{'race': 1}]
privileged_groups_sex = [{'sex': 0}]
unprivileged_groups_sex = [{'sex': 1}]

# make training splits
dataset_orig_train, dataset_orig_test = binaryLabelDataset.split([0.7], shuffle=True)

# compute mean difference on original training data
metric_orig_train_race = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups_race,
                                             privileged_groups=privileged_groups_race)
metric_orig_train_sex = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups_sex,
                                             privileged_groups=privileged_groups_sex)

print("Difference in mean outcomes between unprivileged and privileged groups for race = %f" % metric_orig_train_race.mean_difference())
print("Difference in mean outcomes between unprivileged and privileged groups for sex = %f" % metric_orig_train_sex.mean_difference())

# train
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': dataset_orig_train.instance_weights}
rf_orig = model.fit(dataset_orig_train.features, dataset_orig_train.labels.ravel(), **fit_params)

# thresh_arr = np.linspace(0.01, 0.5, 50)
thresh_arr = [0.3]
val_metrics = test(dataset=dataset_orig_test,
                   model=rf_orig,
                   thresh_arr=thresh_arr,
                   unprivileged_groups=unprivileged_groups_race,
                   privileged_groups=privileged_groups_race)
print(val_metrics)

# mitigate bias on dataset using preprocessing
RW_race = Reweighing(unprivileged_groups=unprivileged_groups_race,
                privileged_groups=privileged_groups_race)
dataset_transf_train_race = RW_race.fit_transform(dataset_orig_train)

RW_sex = Reweighing(unprivileged_groups=unprivileged_groups_sex,
                privileged_groups=privileged_groups_sex)
dataset_transf_train_sex = RW_sex.fit_transform(dataset_orig_train)

# compute mean difference on preprocessing mitigated training data
metric_transf_train_race = BinaryLabelDatasetMetric(dataset_transf_train_race,
                                               unprivileged_groups=unprivileged_groups_race,
                                               privileged_groups=privileged_groups_race)
print("Difference in mean outcomes between unprivileged and privileged groups for race preprocessing = %f" % metric_transf_train_race.mean_difference())

metric_transf_train_sex = BinaryLabelDatasetMetric(dataset_transf_train_sex,
                                               unprivileged_groups=unprivileged_groups_sex,
                                               privileged_groups=privileged_groups_sex)
print("Difference in mean outcomes between unprivileged and privileged groups for sex preprocessing = %f" % metric_transf_train_sex.mean_difference())

# train on mitigated data for race
model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
fit_params = {'randomforestclassifier__sample_weight': dataset_transf_train_race.instance_weights}
rf_mit = model.fit(dataset_transf_train_race.features, dataset_transf_train_race.labels.ravel(), **fit_params)

# thresh_arr = np.linspace(0.01, 0.5, 50)
thresh_arr = [0.3]
val_metrics = test(dataset=dataset_orig_test,
                   model=rf_mit,
                   thresh_arr=thresh_arr,
                   unprivileged_groups=unprivileged_groups_race,
                   privileged_groups=privileged_groups_race)
print(val_metrics)

# graph
# data = [metric_orig_train_race.mean_difference(), metric_orig_train_sex.mean_difference(), metric_transf_train_race.mean_difference(), metric_transf_train_sex.mean_difference()]
# plt.bar(["Race Unmitigated", "Sex Unmitigated", "Race Mitigated", "Sex Mitigated"], data)
# plt.axhline(0, color='black')
# plt.title("Mean Difference")
# plt.show()
