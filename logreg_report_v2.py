#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    log_loss,
    SCORERS, 
    get_scorer,
    classification_report, 
    ConfusionMatrixDisplay, 
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC


# In[ ]:


def make_base(X, y, y_pred):
    base = pd.concat([X, y.to_frame()], axis=1).reset_index()
    base = pd.concat([base, pd.Series(y_pred, name='score').to_frame()], axis=1)
    return base

def gini(target, score):
    try:
        gini = 2*roc_auc_score(target, score)-1
    except ValueError:
#         gini = np.nan
        gini = 0
        pass
    return gini

def gini_stability(X, y, y_pred, w_fallingrate=88.0, w_resstd=-0.5):
    base = make_base(X, y, y_pred)
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
        .sort_values("WEEK_NUM")\
        .groupby("WEEK_NUM")[["target", "score"]]\
        .apply(lambda x: gini(x.target, x.score)).tolist()

#     gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
#         .sort_values("WEEK_NUM")\
#         .groupby("WEEK_NUM")[["target", "score"]]\
#         .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()
    
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

def generate_report(mdl, X, y, cols, extra_metrics=None):
    '''Takes in a [what kind?] model as a string, pandas dataframe of a subset of the test data, 
    and list of columns to be used in the model.
    If necessary, can include a list of other metrics to be outputted, but default is None.
    Returns a dictionary of the following breakdown of scores: AUC, F1-Score, and any extra metrics
    '''
    # Load in a model we have trained and saved somewhere else in the repo
#     model = load(os.path.join(MODELS_DIR, mdl))

    # for now, we will be using a model trained in the notebook
    model = mdl
#     X_test, y_test = df[cols], df["target"]
#     y_pred = model.predict(X)
    y_pred = [probs[1] for probs in model.predict_proba(X)]
    
    res = dict()
    res['AUC'] = roc_auc_score(y, y_pred)
    res['Gini Stability'] = gini_stability(X, y, y_pred)
        
    for metric in extra_metrics:
        if metric == 'log_loss':
            res[metric] = log_loss(y, y_pred)
            print(f"Log Loss: {log_loss(y, y_pred)}")
        else:
            res[metric] = get_scorer(metric)._score_func(y, y_pred)
#         metric_func = getattr(locals()['__builtins__'], metric)
            print(f"{metric}: {get_scorer(metric)._score_func(y, y_pred)}")
        

    # Results
#     print(roc_auc_score(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y, y_pred)}")
    print(f"Gini Stability Score: {gini_stability(X, y, y_pred)}")
        
    return res

