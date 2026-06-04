#!/usr/bin/env python
# coding: utf-8

# # About the problem and our goals
# 
# About the problem
# - Balanced binary classification problem
# - Tiny n and (comparatively) huge p -> Curse of Dimensionality!
# - Potential problems: lots of noise, models might not generalize well, overfitting
# 
# Idea: At first, try to pass the data unfiltered to some baseline models to establish a performance baseline that we can compare to.
# But before we start we encode missing/non-missing information for columns with 40%+ missing values and fill NaN with the feature median.
# 
# For first benchmarking try models like:
# - Tree-Based models, e.g. Random forest
# - Gradient Boosting methods, e.g. XGBoost
# - Logistic Regression, Linear SVC
# 
# Afterwards, we want to find the features that have actual predictive power. For this we do mutliple things:
# - Look at feature variance and mutual information and drop all columns that are constant, nearly constant or add no mutual information
# - Look at feature and permutation importance to further reduce the feature space
# - Look at pairwise correlation and drop columns with high pairwise correlation
# - Look at multicolinearity (VIF) and SHAP values to see if there are still any columns without predictive power left
# 
# After this we should be left with 5-20 columns. With that data, we can train some models and compare performance to the initial models.
# After reducing the feature space in this way other models might also become viable:
# - kNN
# - RBF/Poly SVC
# 
# Apart from standard packages (`numpy`, `pandas`, `sklearn`, ...) you will also need the following packages:
# - `fastparquet` to save/reload output so compute-intensive cells do not need to be run every single time
# - `statsmodels` to calculate the Variance Inflation Factor (VIF)
# - `shap` to look at shap values

# In[1]:


#Importing all the needed libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap

from IPython.core.magic import register_cell_magic
import json

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import (
    KFold,
    cross_val_score,
    GridSearchCV,
    train_test_split
)

from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample


# In[2]:


# Skip cell magic. Annotate a cell with `%%skip` at the top and it will skip.
# Used to skip compute-intensive cells (especially the feature/permutation selection cell that takes 20-30 minutes),
#  but also hyperparameter searches
@register_cell_magic
def skip(line, cell):
    print("-> Skipping this cell.")


# In[12]:


#First step, naive approach
X = pd.read_excel('data/Serma_Pilotstudie.xlsx')
print(f"Data frame has shape {X.shape} meaning {X.shape[0]} points and {X.columns.shape} features!") 

y_label = 'Serom_postoperativ'
Y = X[y_label]
X.drop(y_label, axis=1, inplace=True)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

print(f"Ratio of zeros: {(Y == 0).sum() / Y.shape[0]}, meaning we're working with a balanced dataset!")

#Adds Columns to keep better overview of NaN-Values
for col in X.columns[X.isna().mean() > 0.4]:
    X[f"{col}_missing_na"] = X[col].isna().astype(np.int8)

X = X.fillna(X.median())

print(X.shape)
X


# # Step 1: Establishing some model baselines

# In[3]:


def search_hyperparams(clf, param_grid, X, Y):
    """
    Arguments: Classifier-Modell, Parameter-Grid, X and Y -- Performs GridSearch, meaning testing for the optimal combination of given hyperparameters.
    """
    grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
    grid_search.fit(X, Y)
    
    print(f"Got score {np.round(grid_search.best_score_, 4)} with params {grid_search.best_params_}")


# ## Logistic Regression

# In[4]:


def get_logreg_hyperparams(X, Y):
    """
    Arguments: X and Y, Uses search_hyperparams for Logistic Regression
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=0, max_iter=10000))
    ])
    
    param_grid = {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['liblinear', 'saga'],
    }
    
    search_hyperparams(pipe, param_grid, X, Y)


# In[3]:


get_ipython().run_cell_magic('skip', '', "get_logreg_hyperparams(X, Y)\n# Got score 0.69 with params {'clf__C': 0.1, 'clf__penalty': 'l2', 'clf__solver': 'saga'}\n")


# ## Linear SVC

# In[6]:


def get_lsvc_hyperparams(X, Y, dual=True):
    """
    Arguments: X and Y, Uses search_hyperparams for Linear SVM-Classifier
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(random_state=0))
    ])
    
    param_grid = {
        'clf__penalty': ['l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__loss': ['squared_hinge'],
        # Solve dual problem if p > n - doesn't affect result, just efficiency
        'clf__dual': [dual],
        'clf__max_iter': [100000]
    }
    
    search_hyperparams(pipe, param_grid, X, Y)


# In[4]:


get_ipython().run_cell_magic('skip', '', "get_lsvc_hyperparams(X, Y)\n# Got score 0.67 with params {'clf__C': 0.01, 'clf__dual': True, 'clf__loss': 'squared_hinge', 'clf__max_iter': 100000, 'clf__penalty': 'l2'}\n")


# ## XGBoost

# In[8]:


def get_xgb_hyperparams(X, Y):
    """
    Arguments: X and Y, Uses search_hyperparams for XGBoost Classifier
    """
    param_grid = {
        # Tree complexity
        # Do not set max_depth too high to avoid overfitting
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
    
        # Sampling
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
    
        # Boosting and learning
        'learning_rate': [0.05, 0.1],
        'n_estimators': [10, 20, 50],
    
        # Regularization
        'reg_lambda': [1, 5],
        'reg_alpha': [0, 0.1, 1],
    }

    search_hyperparams(XGBClassifier(random_state=0), param_grid, X, Y)


# In[5]:


get_ipython().run_cell_magic('skip', '', "get_xgb_hyperparams(X, Y)\n# Got score 0.75 with params {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3, 'n_estimators': 20, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.8}\n")


# ## Random Forest

# In[10]:


def get_rfc_hyperparams(X, Y):
    """
    Arguments: X and Y, Uses search_hyperparams for Random Forest Classifier
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [10, 20, 50, 70, 100],
        'max_depth': [2, 3, 4],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    search_hyperparams(RandomForestClassifier(random_state=0), param_grid, X, Y)


# In[6]:


get_ipython().run_cell_magic('skip', '', "get_rfc_hyperparams(X, Y)\n# Got score 0.74 with params {'criterion': 'entropy', 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 70}\n")


# ## Methods to create base models
# 
# Down below, you may find some methods to create the models we test against, so they can be created using their optimal hyperparameters (as determined above)

# In[ ]:


def get_logreg():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.1, max_iter=10000, penalty='l2', solver='saga', random_state=0))
    ])

def get_lsvc():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(C=0.01, dual=True, loss='squared_hinge', max_iter=10000, penalty='l2', random_state=0))
    ])

def get_xgb():
    return XGBClassifier(colsample_bytree=0.8, gamma=0.1, learning_rate=0.1, max_depth=5, min_child_weight=3,
                             n_estimators=20, reg_alpha=0.1, reg_lambda=1, subsample=0.8, random_state=0)

def get_rfc():
    return RandomForestClassifier(criterion='entropy', max_depth=3, max_features='sqrt', min_samples_leaf=1,
                                  min_samples_split=2, n_estimators=70, random_state=0)


# # Section 2: Getting to a reasonable feature space
# 
# Since we have 100 data points and 214(!) features we have to be very careful about overfitting. As a first step, we want to reduce the feature space by dropping columns that do not seem significant.
# 
# For this we look at multiple things:
# 
# - Look at feature variance and mutual information and drop all columns that are constant, nearly constant or add no mutual information
# - Look at feature and permutation importance to further reduce the feature space. More specifically, LogistricRegressionCV with strict L1-penalty to shrink coefficients of non-crucial features to 0, in combination with RFC and XGB to catch nonlinear relationships
# - Look at pairwise correlation and drop columns with high pairwise correlation
# - Look at multicolinearity (VIF) and SHAP values to see if there are still any columns without predictive power left

# ## Dropping constant and near-constant features as well as features with no predictive power

# In[ ]:


print(f'Minimum variance: {X.var(axis=0).min()}')
print(f'Minimum variance (scaled): {MinMaxScaler().fit_transform(X).var(axis=0).min()}')


# In[ ]:


# So it seems there are no almost constant features.
# We do not need to drop any features based on variance alone.
# Next, let's look at mutual information.
def run_mutual_info_reduction():
    mi = mutual_info_classif(X, Y, random_state=0)
    
    no_info_idx = np.where(mi == 0)
    X.drop(X.columns[no_info_idx], axis=1, inplace=True)
    
    print(f'Dropped {len(no_info_idx[0])} columns with no predictive power.')
    print(f'X shape is now {X.shape}.')

    mutual_info_run = True

# THIS CELL SHOULD NOT BE RUN MORE THAN ONCE, OTHERWISE WE DROP UNNECCESSARILY MANY FEATURES
# That's what this safegate is for. Might not be pretty but it works.
try:
    mutual_info_run
except:
    run_mutual_info_reduction()    
else:
    if not mutual_info_run:
        run_mutual_info_reduction()
    else:
        print(f'Mutual info reduction was run already, dataset was already reduced to {X.shape[1]} features!')


# ## Helper methods to look at feature importance
# 
# We use the following three models for this: Random Forest, XGBoost and Logistic Regression.
# 
# The reason why we chose the models is the following:
# - Random Forest chooses optimal splits greedily and has less variance and is more stable than something like Extra Trees
# - XGBoost is regularized, downweights noisy splits, penalizes weak improvements and is good at detecting real signal
# - Logistic Regression is good at finding linear signals
# 
# So if a feature is not used by all three models that's a signal that it's probably just noise and can be dropped safely.

# In[ ]:


def calc_important_features(coef_fn, threshold, X, Y, model_name, plot=False):
    """
    Arguments: 
    - coef_fn: Coefficient function ...
    - Threshold: Value that has to be passed to keep a column
    - X and Y
    - model_name
    """
    n_bootstraps = 100 #what is a bootstrap
    feature_counts = np.zeros(X.shape[1])
    
    for i in range(n_bootstraps):
        X_bs, Y_bs = resample(X, Y, random_state=i) #reshuffle
        feature_counts += coef_fn(X_bs, Y_bs, i) #???
    
    stability = feature_counts / n_bootstraps #???

    if plot: #can be ignored by now
        stability_df = pd.DataFrame({'stability': stability})
        bins = np.linspace(0, 1, 21)
        
        plt.figure(figsize=(8,5))
        sns.histplot(stability_df['stability'], bins=bins, kde=False, color='skyblue')
        plt.xlabel(f'Feature Stability (Selection Frequency - {model_name})')
        plt.ylabel('Number of Features')
        plt.title(f'Feature Stability Distribution - {model_name}')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    X_stable = X.loc[:, stability >= threshold] #New Feature space
    print(f'Kept {X_stable.shape[1]} features ({model_name})!')

    return X_stable

def get_logreg_coef(X_bs, Y_bs, i):
    logreg = LogisticRegressionCV(
        penalty='elasticnet', solver='saga',
        # Keep the model L1-heavy to select only the strongest features
        l1_ratios=[0.8, 0.85, 0.9, 0.95],
        # Allow somewhat weaker penalties
        Cs=[0.1, 1, 10], cv=5,
        max_iter=10000, random_state=i, n_jobs=-1
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', logreg)
    ])

    pipe.fit(X_bs, Y_bs)
    print(f'Run #{i}: {pipe["clf"].C_}')
    return np.abs(pipe['clf'].coef_).ravel() > 1e-3

def get_rfc_coef(X_bs, Y_bs, _):
    rfc = get_rfc()
    rfc.fit(X_bs, Y_bs)
    return np.abs(rfc.feature_importances_).ravel() > 1e-3

def get_xgb_coef(X_bs, Y_bs, i):
    xgb = get_xgb()
    xgb.fit(X_bs, Y_bs)
    return np.abs(xgb.feature_importances_).ravel() > 1e-3

def filter_by_features(X, Y):
    X_stable_logreg = calc_important_features(get_logreg_coef, 0.8, X, Y, 'Logistic Regression')

    X_stable_rfc = calc_important_features(get_rfc_coef, 0.8, X, Y, 'Random Forest')
    X_stable_logreg_rfc = calc_important_features(get_rfc_coef, 0.8, X_stable_logreg, Y, 'Random Forest (based on logreg features)')
    
    # XGB aggressively reduces weights so if a feature is useful in at least 30% of runs that's enough for us.
    X_stable_xgb = calc_important_features(get_xgb_coef, 0.3, X, Y, 'XGBoost')
    X_stable_logreg_xgb = calc_important_features(get_xgb_coef, 0.3, X_stable_logreg, Y, 'XGBoost (based on logreg features)')
    
    # No need to intersect it with X_stable_logreg as both X_stable_logreg_rfc and X_stable_logreg_xgb depend on it

    # Intersect the features that had linear correlation to the target and that both XGB and RFC found interesting.
    X_stable_logreg_final_columns = X_stable_logreg_rfc.columns.intersection(X_stable_logreg_xgb.columns)
    
    X_stable_tree_final_columns = X_stable_rfc.columns.intersection(X_stable_xgb.columns)
    X_stable_final_columns = X_stable_logreg_final_columns.union(X_stable_tree_final_columns)
    
    X_stable = X[X_stable_final_columns].copy()
    print(f'Kept {X_stable.shape[1]} features after looking at different models!')

    return X_stable


# ## Helper methods to look at permutation importance

# In[ ]:


def get_perm_importances(clf, X, Y):
    splits = 10
    kfold = KFold(n_splits=splits, shuffle=True, random_state=0)
    perm_importances = np.empty([splits, X.shape[1]])

    for i, (train_idx, test_idx) in enumerate(kfold.split(X, y=Y)):
        (X_train, Y_train) = (X.iloc[train_idx], Y.iloc[train_idx])
        (X_test, Y_test) = (X.iloc[test_idx], Y.iloc[test_idx])

        c = clone(clf)
        c.fit(X_train, Y_train)

        perm_importances[i, :] = permutation_importance(c, X_test, Y_test, n_repeats=20, random_state=0).importances_mean

    perm_importances = pd.Series(perm_importances.mean(axis=0), index=X.columns)
    return perm_importances

def filter_by_perm_importances(X_stable, Y):
    # At the start, all indices are relevant
    relevant_idx_intersect = pd.Index(X_stable.columns)
    relevant_idx_union = pd.Index([])
    models = [get_logreg(), get_rfc(), get_xgb()]
    
    for i, clf in enumerate(models):
        pi = get_perm_importances(clf, X_stable, Y)
        relevant_idx = pi.index[pi > 0]
    
        relevant_idx_intersect = relevant_idx_intersect.intersection(relevant_idx)
        relevant_idx_union = relevant_idx_union.union(relevant_idx)
        
        print(f'Done with {i + 1}')

    print('Done with all')
    return X_stable[relevant_idx_union].copy()


# ## Looking at feature and permutation importance
# 
# Here we look at feature and permutation importance. For this we invoke the helper methods above mutliple times with different subsets of out data and investigate which features are used consistently across all runs.

# In[ ]:


get_ipython().run_cell_magic('skip', '', "# Skip this cell, you really do not want to run it every time. Takes around 20-30 minutes.\n\nruns = 10\nselected_cols = []\n\nfor i in range(runs):\n    X_train, _, Y_train, _ = train_test_split(X, Y, train_size=80, stratify=Y, random_state=i)\n    X_stable = filter_by_features(X_train, Y_train)\n    X_stable_perm = filter_by_perm_importances(X_stable, Y_train)\n\n    selected_cols.append(set(X_stable_perm.columns))\n    print(f'--- RUN {i+1} ---: Reduced {X.shape[1]} features to {X_stable.shape[1]} to {X_stable_perm.shape[1]}.')\n\n# Save output because this cell takes quite some time to run\nwith open('data/important_cols.json', 'w') as f:\n    json.dump([list(s) for s in selected_cols], f, indent=2)\n")


# In[ ]:


# Reload output so the cell above does not need to be run every time
with open('data/important_cols.json', 'r') as f:
    selected_cols = [set(x) for x in json.load(f)]


# In[ ]:


appearance_count = Counter()
for cols in selected_cols:
    appearance_count.update(cols)

freq_df = pd.DataFrame(list(appearance_count.items()), columns=['col_name', 'freq'])

bins = np.arange(0.5, freq_df['freq'].max()+1.5, 1)

plt.figure(figsize=(8,5))
sns.histplot(freq_df['freq'], bins=bins, kde=False, color='skyblue')

threshold = 5.5
plt.xlabel('Number of runs feature appeared in')
plt.ylabel('Number of features')
plt.title('Feature stability across runs')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')

plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


significant_cols = freq_df[freq_df['freq'] > threshold]['col_name']
X_filtered = X[significant_cols]

print(f'Identified {X_filtered.shape[1]} columns with consistent predictive power!')

X_filtered.to_parquet('data/serma_stable.parquet', engine='fastparquet')


# ## Looking at pairwise correlation
# 
# We've now drastically shrunken the feature space, but maybe there are more features to drop. A first step is to look at pairwise correlation and drop any features that are highly correlated.

# In[ ]:


X_filtered = pd.read_parquet('data/serma_stable.parquet', engine='fastparquet')


# In[ ]:


# Correlation matrix
corr_matrix = pd.concat([Y, X_filtered], axis=1).corr()

plt.figure(figsize=(12,8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
plt.title("Feature Correlation Heatmap (after preprocessing)")
plt.show()


# One can detect the following pairwise correlations:
# - BMI_Ersterhebung and Gewicht_Ersterhebung with 0.88: BMI used in 8 runs and Gewicht only in 7. Furthermore, BMI is probably the better metric, hence we drop Gewicht_Ersterhebung
# - CD163_CD68 and VAR00004 with 0.87: both used in 7 runs. Here we look at feature importances and correlation with the target variable
# - Blut_OP_Granulo and Blut_OP_Lymph with -0.97

# In[ ]:


def get_rfc_xgb_feature_importances(X, Y):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    
    for i, model in enumerate([get_rfc(), get_xgb()]):
        model.fit(X, Y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
        importances.plot.bar(ax=ax[i])
        ax[i].tick_params(axis='x', rotation=90, labelsize=10)
        ax[i].set_yscale('log')
    
    plt.show()


# In[ ]:


# Both models use 'VAR00004' and 'CD163_CD68' roughly equally
get_rfc_xgb_feature_importances(X_filtered, Y)


# In[ ]:


get_ipython().run_cell_magic('skip', '', '# Look at permutation importances to determine which one to drop\n# TODO: There is a minor bug here, the permutation importance changes from run to run.\n#  Probably just a missing random_state somewhere\nrfc_perm_imp = get_perm_importances(get_rfc(), X_filtered, Y)\nxgb_perm_imp = get_perm_importances(get_xgb(), X_filtered, Y)\n\nprint(f\'VAR00004 permutation importance (xgb): {xgb_perm_imp["VAR00004"]}\')\nprint(f\'VAR00004 permutation importance (rfc): {rfc_perm_imp["VAR00004"]}\')\nprint(f\'CD163_CD68 permutation importance (xgb): {xgb_perm_imp["CD163_CD68"]}\')\nprint(f\'CD163_CD68 permutation importance (rfc): {rfc_perm_imp["CD163_CD68"]}\')\nprint(\'---\')\nprint(f\'Blut_OP_Granulo permutation importance (xgb): {xgb_perm_imp["Blut_OP_Granulo"]}\')\nprint(f\'Blut_OP_Granulo permutation importance (rfc): {rfc_perm_imp["Blut_OP_Granulo"]}\')\nprint(f\'Blut_OP_Lymph permutation importance (xgb): {xgb_perm_imp["Blut_OP_Lymph"]}\')\nprint(f\'Blut_OP_Lymph permutation importance (rfc): {rfc_perm_imp["Blut_OP_Lymph"]}\')\n')


# In[ ]:


# Gewicht is dropped as BMI is the better metric
X_filtered.drop('Gewicht_Ersterhebung', axis=1, inplace=True)

# Both have only negative permutation importance but the models use VAR00004 more, hence drop CD163_CD68 (in some runs - see bug above)
X_filtered.drop('CD163_CD68', axis=1, inplace=True)

# Blut_OP_Granulo has the better permutation importance (in some runs - see bug above)
X_filtered.drop('Blut_OP_Lymph', axis=1, inplace=True)


# In[ ]:


def print_cv_scores(X, Y):
    print_cv = lambda clf, clf_name: print(f'{clf_name} crossval score: {np.round(cross_val_score(clf, X, Y, cv=5, scoring="accuracy").mean(), 2)}')
    print_cv(get_rfc(), 'RFC')
    print_cv(get_xgb(), 'XGB')
    print_cv(get_logreg(), 'Logreg')
    print_cv(get_lsvc(), 'Linear SVC')


# In[ ]:


# No significant pairwise correlations left!
corr_matrix = pd.concat([Y, X_filtered], axis=1).corr()
print(f'Maximum (absolute) pairwise correlation: {np.abs(corr_matrix - np.identity(corr_matrix.shape[0])).max().max()}')

print(f'{X_filtered.shape[1]} features left!\n')

# Look at CV scores to get a first feel for how model accuracy might be positively or negatively impacted.
# Of course this can be improved with a new hyperparameter search.
print_cv_scores(X_filtered, Y)


# ## Variance Inflation Factor (VIF)
# 
# Next we look at the variance inflation factor to identify multicolinearity.

# In[ ]:


# Important: Scale X_filtered before calculating VIF
X_filtered_scaled = pd.DataFrame(data=StandardScaler().fit_transform(X_filtered),
                                 index = X_filtered.index, columns=X_filtered.columns)

vif_data = pd.DataFrame({
    'col_name': X_filtered_scaled.columns,
    'VIF': [variance_inflation_factor(X_filtered_scaled.values, i) for i in range(X_filtered_scaled.shape[1])]
})

# Good news, multicolinearity is not a problem!
vif_data.sort_values('VIF', ascending=False)


# ## SHAP values
# 
# And finally, we look at SHAP values to determine which features add value and if a few more can be safely dropped.

# In[ ]:


# Only run this for rfc, as it seems to be the most accurate model
rfc = get_rfc()
rfc.fit(X_filtered, Y)

shap_class1 = shap.TreeExplainer(rfc).shap_values(X_filtered)[:, :, 1]
shap.summary_plot(shap_class1, X_filtered, max_display=13)

mean_abs_shap = pd.Series(np.abs(shap_class1).mean(axis=0), index=X_filtered.columns).sort_values(ascending=False)

print(mean_abs_shap)
print(f'Lowest (absolute) shap value is {np.round(mean_abs_shap.min(), 4)} which is more than 10% of the highest value {np.round(mean_abs_shap.max(), 4)}')
print('We will therefore keep all values')


# In[ ]:


# One could think about dropping "Anzahl_entnommene_LK" and "Stillen" as they don't seem to contribute that much
get_rfc_xgb_feature_importances(X_filtered, Y)
# And both RFC and XGB don't weigh them very much


# In[ ]:


# To investigate whether we can safely drop them, we compare model scores with and without them
X_filtered_slim = X_filtered.drop(['Anzahl_entnommene_LK', 'Stillen'], axis=1, inplace=False)

print('Full feature set cv scores')
print_cv_scores(X_filtered, Y)
print('\nReduced feature set cv scores')
print_cv_scores(X_filtered_slim, Y)

# Model score doesn't change really, the features can be dropped
X_final = X_filtered_slim


# In[ ]:


print(f"We're left with {X_final.shape[1]} final features!")
X_final.to_parquet('data/serma_final.parquet', engine='fastparquet')


# # Section 3: Comparing model scores
# 
# In this section we look at how model accuracy changes due to our feature selection.
# We'll compare all four models (Logreg, RFC, XGB, LSVC) to their baseline as well as evaluate some new models.

# In[ ]:


X_final = pd.read_parquet('data/serma_final.parquet', engine='fastparquet')


# ## Comparing model accuracy to baseline score

# In[ ]:


def compare_model_performance(eval_fn):
    print(f'--- Baseline model score with full {X.shape[1]} features ---')
    eval_fn(X, Y)
    print(f'--- Improved model score with reduced set of {X_final.shape[1]} features ---')
    eval_fn(X_final, Y)


# In[ ]:


get_ipython().run_cell_magic('skip', '', "compare_model_performance(get_logreg_hyperparams)\n# --- Baseline model score with full 117 features ---\n# Got score 0.7 with params {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'liblinear'}\n# --- Improved model score with reduced set of 10 features ---\n# Got score 0.78 with params {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'liblinear'}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "compare_model_performance(get_rfc_hyperparams)\n# --- Baseline model score with full 117 features ---\n# Got score 0.77 with params {'criterion': 'gini', 'max_depth': 2, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}\n# --- Improved model score with reduced set of 10 features ---\n# Got score 0.81 with params {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 20}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "compare_model_performance(get_xgb_hyperparams)\n# --- Baseline model score with full 117 features ---\n# Got score 0.76 with params {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'n_estimators': 10, 'reg_alpha': 0, 'reg_lambda': 5, 'subsample': 0.8}\n# --- Improved model score with reduced set of 10 features ---\n# Got score 0.82 with params {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 50, 'reg_alpha': 0.1, 'reg_lambda': 1, 'subsample': 0.8}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "print(f'--- Baseline model score with full {X.shape[1]} features ---')\nget_lsvc_hyperparams(X, Y, dual=True)\nprint(f'--- Improved model score with reduced set of {X_final.shape[1]} features ---')\nget_lsvc_hyperparams(X_final, Y, dual=False)\n# --- Baseline model score with full 117 features ---\n# Got score 0.66 with params {'clf__C': 10, 'clf__dual': True, 'clf__loss': 'squared_hinge', 'clf__max_iter': 100000, 'clf__penalty': 'l2'}\n# --- Improved model score with reduced set of 10 features ---\n# Got score 0.78 with params {'clf__C': 0.01, 'clf__dual': False, 'clf__loss': 'squared_hinge', 'clf__max_iter': 100000, 'clf__penalty': 'l2'}\n")


# ## Evaluating new models: KNN and SVC

# In[ ]:


def get_knn_hyperparams(X, Y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])
    
    param_grid = {
        'clf__n_neighbors': [2, 3, 5, 7],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2]
    }

    search_hyperparams(pipe, param_grid, X, Y)


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_knn_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__n_neighbors': 5, 'clf__p': 1, 'clf__weights': 'uniform'}\n")


# In[ ]:


# Gamma value for gamma=scale for the svm
1.0 / (StandardScaler().fit_transform(X_final).var() * X_final.shape[1])


# In[ ]:


def get_svc_hyperparams(X, Y):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(random_state=0))
    ])
    
    param_grid = [
        {
            'clf__C': [1, 5, 10, 15],
            'clf__kernel': ['rbf'],
            'clf__gamma': ['scale', 0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
        },
        {
            'clf__C': [1, 5, 10, 15],
            'clf__kernel': ['poly'],
            'clf__degree': [2, 3, 4],
            'clf__gamma': ['scale', 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, 0.2]
        }]

    search_hyperparams(pipe, param_grid, X, Y)


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_svc_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}\n")


# # Looking at PCA
# 
# We have a pretty minimal set of features, but maybe we can reduce dimension even further. To do this we look at the eigenvalues of the covariance matrix and run a hyperparameter search for kNN after PCA preprocessing.
# As Tree-based models (RFC, XGB) are not expected to be influenced much by PCA (unless you reduce the dimension too much) we only look at the other models in the hopes of maybe being able to reduce the dimension even further.

# In[ ]:


X_final_center = StandardScaler().fit_transform(X_final)

cov = np.cov(X_final_center, rowvar=False)
(eigval, eigvec) = np.linalg.eigh(cov)
print('Covariance matrix eigenvalues')
np.sort(eigval)[::-1]


# In[ ]:


def get_pca_knn_hyperparams(X, Y, highest_pca_dim=9):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(copy=True, whiten=False, random_state=0)),
        ('clf', KNeighborsClassifier())
    ])
    
    param_grid = {
        'clf__n_neighbors': [2, 3, 5, 7],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2],
        'pca__n_components': range(1, highest_pca_dim + 1)
    }

    search_hyperparams(pipe, param_grid, X, Y)

def get_pca_logreg_hyperparams(X, Y, highest_pca_dim=9):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(copy=True, whiten=False, random_state=0)),
        ('clf', LogisticRegression(random_state=0, max_iter=10000))
    ])
    
    param_grid = {
        'clf__penalty': ['l1', 'l2'],
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__solver': ['liblinear', 'saga'],
        'pca__n_components': range(1, highest_pca_dim + 1)
    }
    
    search_hyperparams(pipe, param_grid, X, Y)

def get_pca_svc_hyperparams(X, Y, highest_pca_dim=9):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(copy=True, whiten=False, random_state=0)),
        ('clf', SVC(random_state=0))
    ])
    
    param_grid = [
        {
            'clf__C': [1, 5, 10, 15],
            'clf__kernel': ['rbf'],
            'clf__gamma': ['scale', 0.01, 0.03, 0.05, 0.07, 0.1, 0.2],
            'pca__n_components': range(1, highest_pca_dim + 1)
        },
        {
            'clf__C': [1, 5, 10, 15],
            'clf__kernel': ['poly'],
            'clf__degree': [2, 3, 4],
            'clf__gamma': ['scale', 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.12, 0.15, 0.2],
            'pca__n_components': range(1, highest_pca_dim + 1)
        }]

    search_hyperparams(pipe, param_grid, X, Y)

def get_pca_lsvc_hyperparams(X, Y, highest_pca_dim=9):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(copy=True, whiten=False, random_state=0)),
        ('clf', LinearSVC(random_state=0, dual=False, probability=True))
    ])
    
    param_grid = {
        'clf__penalty': ['l2'],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__loss': ['squared_hinge'],
        'clf__max_iter': [100000],
        'pca__n_components': range(1, highest_pca_dim + 1)
    }
    
    search_hyperparams(pipe, param_grid, X, Y)


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_pca_knn_hyperparams(X_final, Y)\nget_knn_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__n_neighbors': 5, 'clf__p': 2, 'clf__weights': 'uniform', 'pca__n_components': 6}\n# Got score 0.8 with params {'clf__n_neighbors': 5, 'clf__p': 1, 'clf__weights': 'uniform'}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_pca_logreg_hyperparams(X_final, Y, 9)\nget_pca_logreg_hyperparams(X_final, Y, 6)\nget_logreg_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__C': 0.1, 'clf__penalty': 'l2', 'clf__solver': 'liblinear', 'pca__n_components': 7}\n# Got score 0.8 with params {'clf__C': 1, 'clf__penalty': 'l1', 'clf__solver': 'liblinear', 'pca__n_components': 6}\n# Got score 0.78 with params {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'liblinear'}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_pca_lsvc_hyperparams(X_final, Y)\nget_lsvc_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__C': 0.01, 'clf__loss': 'squared_hinge', 'clf__max_iter': 100000, 'clf__penalty': 'l2', 'pca__n_components': 6}\n# Got score 0.78 with params {'clf__C': 0.01, 'clf__dual': True, 'clf__loss': 'squared_hinge', 'clf__max_iter': 100000, 'clf__penalty': 'l2'}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "get_pca_svc_hyperparams(X_final, Y)\nget_svc_hyperparams(X_final, Y)\n# Got score 0.8 with params {'clf__C': 1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf', 'pca__n_components': 6}\n# Got score 0.8 with params {'clf__C': 10, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}\n")


# In[ ]:


get_ipython().run_cell_magic('skip', '', "# It seems that the space could be safely reduced to dimension six\n# But even more surprising:\nget_pca_logreg_hyperparams(X_final, Y, 1)\nget_pca_svc_hyperparams(X_final, Y, 1)\n# Got score 0.78 with params {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'saga', 'pca__n_components': 1}\n# Got score 0.79 with params {'clf__C': 1, 'clf__gamma': 0.07, 'clf__kernel': 'rbf', 'pca__n_components': 1}\n\n# It seems that for some models we can safely reduce the space to one dimension and lose only insignificant prediction power!\n")


# # Summary
# 
# By looking at mutual information, feature and permutation importance, pairwise correlation, VIF and SHAP values we were able to reduce the original 133 features to just 10 features with real predictive power. These features are (without any order):
# - OP_Gewicht_Präp
# - Horm_KC_Dauer
# - Alter_OP
# - Blut_OP_Granulo
# - BMI_Ersterhebung
# - CD163_AT
# - OP_Revision_missing_na
# - PR_post_OP_IRS
# - VAR00004
# - OP_ax_Eingriff
# 
# By testing with KNN, (Linear and Nonlinear) SVC and Logistic Regression we determined that the feature space can be reduced by an additional 40% (so 6 dimensions remaining) via PCA without losing predictive power. In some cases (RBF SVC, Logistic Regression) we can even use PCA to reduce the problem to one dimension without losing much (only 1-2%) predictive power.
# 
# After feature reduction all our models received a score in the 78-82% range which is an average improvement of ~8%, 6% if you don't count the Linear SVC which was probably not a good model for the original dataset.
# Since this increase is independent of the chosen model and for a wide range of models (RFC, XGB, Linear and Nonlinear SVC, Logistic Regression, KNN) this signals we've identified features with real predictive power.

# # What could be improved, what could be done in the future
# 
# ## Cutoff values in feature importance
# 
# The cutoff values for when a feature is important vs. for when it is not were chosen somewhat arbitrarily (by looking at plots). A possible improvement might be choosing the top X features or choosing the cutoff value adaptively in relation to the most (or least) significant value.
# 
# ## Comparing our feature selection with prominent algorithms
# 
# It would be interesting to compare our identified features with prominent feature selection algorithms such as Boruta or Recursive Feature Elimination (RFE).

# In[ ]:




