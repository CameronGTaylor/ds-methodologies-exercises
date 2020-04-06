from pydataset import data
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error as mse

import wrangle, split_scale

def plot_residuals(y, yhat):
    resids = pd.DataFrame([yhat - y])
    resids = resids.T
    plt.scatter(x=y, y=resids)
    plt.show()

def regression_errors(y, yhat):
    return {
        'ESS' : sum((yhat - y.mean())**2),
        'TSS' : sum((y - y.mean())**2),
        'MSE' : mse(y, yhat),
        'SSE' : mse(y, yhat) * len(y),
        'RMSE' : sqrt(mse(y, yhat)),
    }
    # print(f'SSE: {SSE}\nESS: {ESS}\nTSS: {TSS}\nMSE: {MSE}\nRMSE: {RMSE}')
    # return TSS, ESS, SSE, MSE, RMSE

def baseline_mean_errors(y):
    df_ = pd.DataFrame(index=y, columns=['n'])
    df_ = df_.fillna(y.mean())
    return {
        'MSE' : mse(y, df_),
        'SSE' : mse(y, df_ * len(y)),
        'RMSE' : sqrt(mse(y, df_)),
    }
    # print(f'SSE: {SSE}\nMSE: {MSE}\nRMSE: {RMSE}')
    # return SSE, MSE, RMSE

def better_than_baseline(y, yhat):
    MSE1 = regression_errors(y, yhat).get('MSE')
    MSE2 = baseline_mean_errors(y).get('MSE')
    if MSE1 < MSE2:
        print(f'True: {(MSE2 - MSE1):.2f} is greater than 0')
    else:
        print(f'False: {(MSE2 - MSE1):.2f} is less than 0')

def model_significance(model):
    return model.rsquared, model.f_pvalue