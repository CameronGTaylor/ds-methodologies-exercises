import pandas as pd
import numpy as np
import sklearn.preprocessing as skl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import wrangle

def split_my_data(df, train_pct):
    train, test = (
        train_test_split(df, train_size=train_pct,
            random_state=42))
    return train, test

def split_my_data_twice(X, y, train_pct):
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, train_size = train_pct,
                             random_state=42))
    return X_train, X_test, y_train, y_test

def standard_scaler(train, test):
    scaler = skl.StandardScaler().fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), 
                    columns=train.columns, index=train.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test), 
                    columns=test.columns, index=test.index)
    
    return scaler, train_scaled, test_scaled

def scale_inverse(scaler, train_scaled, test_scaled):
    train = pd.DataFrame(scaler.inverse_transform(train_scaled),
        columns = train_scaled.columns, index = train_scaled.index)
    
    test = pd.DataFrame(scaler.inverse_transform(test_scaled),
        columns = test_scaled.columns, index = test_scaled.index)
    
    return train, test

def uniform_scaler(train, test):
    scaler = skl.QuantileTransformer(n_quantiles=100,
                                output_distribution='uniform')
    scaler.fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), 
                    columns=train.columns, index=train.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test), 
                    columns=test.columns, index=test.index)
    
    return scaler, train_scaled, test_scaled

def gaussian_scaler(train, test):
    scaler = skl.PowerTransformer(method='yeo-johnson')
    scaler.fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), 
                    columns=train.columns, index=train.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test), 
                    columns=test.columns, index=test.index)
    
    return scaler, train_scaled, test_scaled

def min_max_scaler(train, test):
    scaler = skl.MinMaxScaler()
    scaler.fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), 
                    columns=train.columns, index=train.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test), 
                    columns=test.columns, index=test.index)
    
    return scaler, train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = skl.RobustScaler(quantile_range=(25.0,75.0))
    scaler.fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), 
                    columns=train.columns, index=train.index)
    
    test_scaled = pd.DataFrame(scaler.transform(test), 
                    columns=test.columns, index=test.index)
    
    return scaler, train_scaled, test_scaled


