# Data_Imputation_COMPAS

The experiment can be found in run_experiment.py. To run it for each method (baseline, row deletion, predictive mean matching, MissForest) the corresponding data is inputed/uncommented.

Initial data processing was done in data/preprocessing.py, and the creation of the inputed data is done in data/run_pmm.py and data/run_row_deletion.py. MissForest data was done using the MissForest R package found [here](https://github.com/stekhoven/missForest) and is therefore not included in this file structure.
