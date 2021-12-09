import sys
import numpy as np
import pandas as pd
import aif360
from aif360 import datasets
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import StandardScaler
from IPython.display import Markdown, display
from matplotlib import pyplot as plt

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

# make training splits
dataset_orig_train, dataset_orig_test = binaryLabelDataset.split([0.7], shuffle=True)

# compute fairness on original data
privileged_groups_race = [{'race': 0}]
unprivileged_groups_race = [{'race': 1}]
privileged_groups_sex = [{'sex': 0}]
unprivileged_groups_sex = [{'sex': 1}]

metric_orig_train_race = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups_race,
                                             privileged_groups=privileged_groups_race)
metric_orig_train_sex = BinaryLabelDatasetMetric(dataset_orig_train,
                                             unprivileged_groups=unprivileged_groups_sex,
                                             privileged_groups=privileged_groups_sex)

print("Difference in mean outcomes between unprivileged and privileged groups for race = %f" % metric_orig_train_race.mean_difference())
print("Difference in mean outcomes between unprivileged and privileged groups for sex = %f" % metric_orig_train_sex.mean_difference())

# mitigate bias on dataset using preprocessing
RW_race = Reweighing(unprivileged_groups=unprivileged_groups_race,
                privileged_groups=privileged_groups_race)
dataset_transf_train_race = RW_race.fit_transform(dataset_orig_train)

RW_sex = Reweighing(unprivileged_groups=unprivileged_groups_sex,
                privileged_groups=privileged_groups_sex)
dataset_transf_train_sex = RW_sex.fit_transform(dataset_orig_train)

# compute fairness on preprocessing mitigated dataset
metric_transf_train_race = BinaryLabelDatasetMetric(dataset_transf_train_race,
                                               unprivileged_groups=unprivileged_groups_race,
                                               privileged_groups=privileged_groups_race)
print("Difference in mean outcomes between unprivileged and privileged groups for race preprocessing = %f" % metric_transf_train_race.mean_difference())

metric_transf_train_sex = BinaryLabelDatasetMetric(dataset_transf_train_sex,
                                               unprivileged_groups=unprivileged_groups_sex,
                                               privileged_groups=privileged_groups_sex)
print("Difference in mean outcomes between unprivileged and privileged groups for sex preprocessing = %f" % metric_transf_train_sex.mean_difference())

# graph
data = [metric_orig_train_race.mean_difference(), metric_orig_train_sex.mean_difference(), metric_transf_train_race.mean_difference(), metric_transf_train_sex.mean_difference()]
plt.bar(["Race Unmitigated", "Sex Unmitigated", "Race Mitigated", "Sex Mitigated"], data)
plt.axhline(0, color='black')
plt.title("Disparate Impact")
plt.show()

# # mitigate bias on dataset using DisparateImpactRemover
# di = DisparateImpactRemover(repair_level = 1.0)
# dataset_di_train = di.fit_transform(dataset_orig_train)
#
# # compute fairness on DisparateImpactRemover mitigated dataset
# metric_transf_train_di_race = BinaryLabelDatasetMetric(dataset_di_train,
#                                                unprivileged_groups=unprivileged_groups_race,
#                                                privileged_groups=privileged_groups_race)
# print("Difference in mean outcomes between unprivileged and privileged groups for race DisparateImpactRemover = %f" % metric_transf_train_di_race.mean_difference())
#
# metric_transf_train_di_sex = BinaryLabelDatasetMetric(dataset_di_train,
#                                                unprivileged_groups=unprivileged_groups_sex,
#                                                privileged_groups=privileged_groups_sex)
# print("Difference in mean outcomes between unprivileged and privileged groups for sex DisparateImpactRemover = %f" % metric_transf_train_di_sex.mean_difference())
