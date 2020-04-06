import pandas as pd
import numpy as np
from pydataset import data

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

import split_scale

def select_kbest(X, y, k):
    selector = SelectKBest(f_regression, k).fit(X, y)
    X_kbest = selector.transform(X)
    support = selector.get_support()
    best_features = X.columns[support]
    return X_kbest, best_features

def select_rfe(X, y, k):
    rfe = RFE(LinearRegression(), k)
    X_rfe = rfe.fit_transform(X, y)
    mask = rfe.support_
    best_features = X.columns[mask]
    return X_rfe, best_features