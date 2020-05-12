import pandas as pd
import requests
import acquire
import matplotlib.pyplot as plt
import seaborn as sns

def make_datetime(df):
    df.sale_date = pd.to_datetime(df['sale_date'], format='%a, %d %b %Y %H:%M:%S %Z')
    return df

def plot_sales_vs_amount(df):
    plt.rc('figure', figsize=(16,9))
    sns.scatterplot(df.sale_amount,df.item_price)
    plt.show()

def set_index(df):
    df.set_index('sale_date', inplace=True)
    return df

def add_columns(df):
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['sales_total'] = df.sale_amount * df.item_price
    df["sales_diff"] = df.sales_total.diff(1)
    return df