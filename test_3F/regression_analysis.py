import os
import numpy as np
from test_3F.utils import DATASET_DIR, RESULTS_DIR
import pandas as pd
import warnings
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

#%% Load and prepare data
def load_dataset():
    dataset = pd.read_csv(os.path.join(DATASET_DIR, "features_dataset.csv"), sep=";", index_col=0)
    subjective_score = pd.read_csv(os.path.join(DATASET_DIR, "subjective_scores.csv"), sep=";", index_col=0)["DXcelkem"]

    # Filter out subjects that are missing from subjective_score
    missing_subjects = [subject for subject in dataset.index.tolist() if subject not in subjective_score.index.tolist()]
    dataset.drop(labels=missing_subjects, axis=0, inplace=True)

    # Sort dataset's indexes
    if dataset.index.tolist() is not subjective_score.index.tolist():
        dataset.sort_index(inplace=True)
        subjective_score.sort_index(inplace=True)

    # Impute missing values
    dataset.fillna(dataset.median(axis=0), inplace=True)
    return dataset, subjective_score


# Create a dict of standard models to evaluate {name:object}
def get_models(models=dict()):
    # linear models
    models['lr'] = LinearRegression()
    alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for a in alpha:
        models['lasso-' + str(a)] = Lasso(alpha=a)
    for a in alpha:
        models['ridge-' + str(a)] = Ridge(alpha=a)
    for a1 in alpha:
        for a2 in alpha:
            name = 'en-' + str(a1) + '-' + str(a2)
            models[name] = ElasticNet(a1, a2)
    models['huber'] = HuberRegressor()
    models['lars'] = Lars()
    models['llars'] = LassoLars()
    models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
    models['ranscac'] = RANSACRegressor()
    models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
    models['theil'] = TheilSenRegressor()

    # non-linear models
    n_neighbors = range(1, 21)
    for k in n_neighbors:
        models['knn-' + str(k)] = KNeighborsRegressor(n_neighbors=k)
    models['cart'] = DecisionTreeRegressor()
    models['extra'] = ExtraTreeRegressor()
    models['svml'] = SVR(kernel='linear')
    models['svmp'] = SVR(kernel='poly')
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for c in c_values:
        models['svmr' + str(c)] = SVR(C=c)
    # ensemble models
    n_trees = [50, 100, 500, 1000, 5000]
    for k in n_trees:
        models['ada' + str(k)] = AdaBoostRegressor(n_estimators=k)
        models['bag' + str(k)] = BaggingRegressor(n_estimators=k)
        models['rf' + str(k)] = RandomForestRegressor(n_estimators=k)
        models['et' + str(k)] = ExtraTreesRegressor(n_estimators=k)
        models['gbm' + str(k)] = GradientBoostingRegressor(n_estimators=k)

    print('Defined %d models' % len(models))
    return models


# Create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline


# Evaluate a single model
def evaluate_model(X, y, model, folds, metric):
    # create the pipeline
    pipeline = make_pipeline(model)
    # evaluate model
    scores = cross_validate(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1, return_estimator=True)
    return scores


# Evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
    scores = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = evaluate_model(X, y, model, folds, metric)
    except:
        scores = None
    return scores


# Evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='neg_mean_squared_error'):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        scores = robust_evaluate_model(X, y, model, folds, metric)
        if scores is not None:
            # store a result
            results[name] = scores
        else:
            print('>%s: error' % name)
    return results


# Print and plot the top n results, return top result
def summarize_results(results, maximize=True, top_n=10, sort_by_metric='neg_mean_squared_error'):
    sort_by_metric = "test_"+sort_by_metric
    # check for no results
    if len(results) == 0:
        print('no results')
        return
    # determine how many results to summarize
    n = min(top_n, len(results))
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k, mean(v[sort_by_metric])) for k, v in results.items()]

    # sort tuples by mean score, descending order if maximize=True
    mean_scores = sorted(mean_scores, key=lambda x: x[1], reverse=maximize)

    # retrieve the top n for summarization
    names = [model[0] for model in mean_scores[:n]]
    scores = [results[name][sort_by_metric] for name in names[:n]]

    # print the top n
    for i in range(n):
        name = names[i]
        mean_score, std_score = mean(results[name][sort_by_metric]), std(results[name][sort_by_metric])
        print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i + 1, name, mean_score, std_score))
    # boxplot for the top n
    plt.boxplot(scores, labels=names)
    _, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    # Save figure
    file_path = os.path.join(RESULTS_DIR, "spotcheck.png")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)

    return results[names[0]], names[0]


def plot_model_results(X, top_result, name, metric):
    # Get metrics
    metric = ["test_"+m for m in metric]
    top_result_metrics = pd.DataFrame(top_result)[metric]
    top_result_metrics.columns = ["R^2", "neg. MAE", "neg. MSE"]
    top_result_metrics_mean = top_result_metrics.mean(axis=0)
    # Get feature importance
    feature_importances = np.empty([len(X.columns.tolist()), len(top_result['estimator'])])
    for i, estimator in enumerate(top_result['estimator']):
        feature_importances[:, i] = estimator.named_steps["model"].coef_
    feature_importances_mean = np.nanmean(feature_importances, axis=1)
    feature_importances_mean = pd.DataFrame(feature_importances_mean, index=X.columns.tolist(), columns=['mean'])
    feature_importances = pd.DataFrame(feature_importances, index=X.columns.tolist(),
                                       columns=['fold_'+str(k+1) for k in range(len(top_result['estimator']))])
    feature_importances = pd.concat([feature_importances, feature_importances_mean], axis=1).sort_values(["mean"], key=lambda col: np.abs(col), ascending=False)

    # Save plot to pdf
    file_path = os.path.join(RESULTS_DIR, "spotcheck_top_result.pdf")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with PdfPages(file_path) as pdf:
        plt.style.use('seaborn')
        fig, ax = plt.subplots(2,1, figsize=(8.27, 11.69), gridspec_kw={'height_ratios': [0.2, 9]})
        for a in ax:
            a.axis('tight')
            a.axis('off')
        ax[0].set_title(f"Cross-validation metrics for top model: {name}", pad=20)
        ax[1].set_title(f"Feature importances (sorted by absolute mean importance)", pad=10)
        # Plot model metrics
        table_metrics = ax[0].table(cellText=np.around(top_result_metrics_mean.values.reshape([1,3]),4),
                                    colLabels=top_result_metrics_mean.index, loc='upper center', cellLoc="center")
        table_metrics.auto_set_column_width(col=list(range(len(top_result_metrics_mean.index))))
        table_metrics.set_fontsize(11)
        table_metrics.scale(1, 1.2)
        # Plot feature importance
        table_importances = ax[1].table(cellText=np.around(feature_importances.values,2),
                                        colLabels=feature_importances.columns, rowLabels=feature_importances.index,
                                        loc='upper right', cellLoc="center")
        table_importances.auto_set_column_width(col=list(range(len(feature_importances.columns))))
        table_importances.set_fontsize(11)
        table_importances.scale(1, 1.15)
        pdf.savefig()
        plt.close()


# %%
# load dataset
X,y = load_dataset()
# get model list
models = get_models()
# evaluate models
results = evaluate_models(X, y, models, folds=10, metric=["r2", "neg_mean_absolute_error", 'neg_mean_squared_error'])
# summarize results
top_result, name = summarize_results(results, maximize=True, top_n=10, sort_by_metric="neg_mean_squared_error")
# save cross validation results for top model
plot_model_results(X, top_result, name, metric=["r2", "neg_mean_absolute_error", 'neg_mean_squared_error'])
