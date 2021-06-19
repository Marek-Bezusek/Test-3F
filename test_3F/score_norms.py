import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from test_3F.utils import DATASET_DIR, RESULTS_DIR
import os
import numpy as np

#%% Load and prepare data
dataset = pd.read_csv(os.path.join(DATASET_DIR, "features_dataset.csv"), sep=";", index_col=0)
subjective_score = pd.read_csv(os.path.join(DATASET_DIR, "subjective_scores.csv"), sep=";", index_col=0)[["fonorespirace2","fonetika3","DXcelkem"]]

# Filter out subjects that are missing from subjective_score
missing_subjects = [subject for subject in dataset.index.tolist() if subject not in subjective_score.index.tolist()]
dataset.drop(labels=missing_subjects, axis=0, inplace=True)

# Sort dataset's indexes
if dataset.index.tolist() is not subjective_score.index.tolist():
    dataset.sort_index(inplace=True)
    subjective_score.sort_index(inplace=True)

#%% Split subjective_score by women(1), men(2)
group1, group2 = subjective_score.groupby(by=lambda x: x[1])
subjective_score_women = group1[1]
subjective_score_men = group2[1]

#%% Norm men
f2_men = 27
f3_men = 26
DX_men = 79
# Select subjects within norm
subjective_score_men = subjective_score_men[(subjective_score_men.fonorespirace2 >= f2_men)
                                            & (subjective_score_men.fonetika3 >= f3_men)
                                            & (subjective_score_men.DXcelkem >= DX_men)]

#%% Norm women
f2_women = 25
f3_women = 28
DX_women = 80
# Select subjects within norm
subjective_score_women = subjective_score_women[(subjective_score_women.fonorespirace2 >= f2_women)
                                                & (subjective_score_women.fonetika3 >= f3_women)
                                                & (subjective_score_women.DXcelkem >= DX_women)]

#%% Split feature dataset by women(1), men(2) and filter by norm
group1, group2 = dataset.groupby(by=lambda x: x[1])
dataset_women = group1[1].loc[subjective_score_women.index.tolist(), :]
dataset_men = group2[1].loc[subjective_score_men.index.tolist(), :]

#%% Get feature stats
women_stats = dataset_women.describe(include='all', percentiles=[0.05, .25, .5, .75, .95]).transpose()
men_stats = dataset_men.describe(include='all', percentiles=[0.05, .25, .5, .75, .95]).transpose()
women_stats.drop(labels="count", axis=1, inplace=True)
men_stats.drop(labels="count", axis=1, inplace=True)

#%% Plot and save result
file_path = os.path.join(RESULTS_DIR, "score_norms.pdf")
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with PdfPages(file_path) as pdf:
    for i, stats in enumerate([women_stats, men_stats]):
        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('tight')
        ax.axis('off')
        if i == 0:
            title = "women"
        else:
            title = "men"
        ax.set_title(f"Score norms {title}", pad=20)
        table = ax.table(cellText=np.around(stats.values, 2), colLabels=stats.columns,
                         rowLabels=stats.index, loc='upper right', cellLoc="center")
        table.auto_set_column_width(col=list(range(len(stats.columns))))
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig()
        plt.close()
