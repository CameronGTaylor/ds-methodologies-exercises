import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import floor 
import wrangle, split_scale

def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg',
            plot_kws = {'line_kws': {'color' : 'orange'}})

def months_to_years(months_col, df):
    df['tenure_years'] = (months_col/12).astype(int)
    return df

def plot_categorical_and_continous_vars(
            categorical_var, continuous_var, df):
    x = categorical_var
    y = continuous_var
    data = df
    f, axes = plt.subplots(1,3, sharey=True, figsize=(16,9))
    plt.rc('font', size=12)

    sns.swarmplot(x=x, y=y, data=data, ax=axes[0])
    sns.boxplot(x=x, y=y, data=data, ax=axes[1])
    sns.barplot(x=x, y=y, data=data, ax=axes[2])