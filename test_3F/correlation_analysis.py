from scipy.stats import spearmanr
import pandas as pd
from test_3F.utils import DATASET_DIR, RESULTS_DIR
import os
import matplotlib.pyplot as plt
import numpy as np

#%% Load and prepare data
dataset = pd.read_csv(os.path.join(DATASET_DIR, "features_dataset.csv"), sep=";", index_col=0)
subjective_score = pd.read_csv(os.path.join(DATASET_DIR, "subjective_scores.csv"), sep=";", index_col=0)

# Filter out subjects that are missing from subjective_score
missing_subjects = [subject for subject in dataset.index.tolist() if subject not in subjective_score.index.tolist()]
dataset.drop(labels=missing_subjects, axis=0, inplace=True)

# Sort dataset's indexes
if dataset.index.tolist() is not subjective_score.index.tolist():
    dataset.sort_index(inplace=True)
    subjective_score.sort_index(inplace=True)

# Impute missing values
dataset.fillna(dataset.median(axis=0), inplace=True)

#%% Compute correlation
corr = pd.DataFrame(index=dataset.columns.tolist(), columns=["rho", "p"], dtype='float')
for feature in dataset.columns.tolist():
    task = feature.split("_")[1]
    corr.loc[feature, "rho"], corr.loc[feature, "p"] = spearmanr(dataset[feature].to_numpy(dtype='float'),
                                                                 subjective_score["o"+task].to_numpy(dtype='int'))

#%% Plot and save correlation result
alpha = 0.05
plt.style.use('seaborn')
ig, ax = plt.subplots(figsize=(8, 10))
ax.axis('tight')
ax.axis('off')
plt.title("Spearman correlation of features with subjective scores", pad=30)
table = ax.table(cellText=np.around(corr.values,5), colLabels=corr.columns, rowLabels=corr.index, loc='center',
                 cellLoc="center", bbox=[.3, 0, .5, 1], colWidths=[0.25, 0.25])
table.set_fontsize(11)
# Features with p < alpha in bold
for (row, col), cell in table.get_celld().items():
    if row > 0 and col == 1 and float(cell.get_text().get_text()) < alpha:
        table[row, 0].get_text().set_weight('bold')
        table[row, 1].get_text().set_weight('bold')
        table[row, -1].get_text().set_weight('bold')
# Save figure
file_path = os.path.join(RESULTS_DIR, 'feature_correlation.pdf')
os.makedirs(os.path.dirname(file_path), exist_ok=True)
plt.savefig(file_path)

