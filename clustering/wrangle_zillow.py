import env
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def get_zillow_data():
    url = env.get_db_url('zillow')
    query = '''
    SELECT * FROM predictions_2017
    LEFT JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE (latitude IS NOT NULL AND 
            longitude IS NOT NULL)'''
    df = pd.read_sql(query, url)

    new_dates = df.groupby(by='parcelid').transactiondate.max().reset_index()
    df.drop(columns=['parcelid','transactiondate'], inplace=True)
    df = new_dates.join(df, how='left')
    df.drop(columns='id', inplace=True)
    return df

def null_rows_info(df):
    summary = pd.DataFrame(df.columns)
    summary['num_rows_missing'] = df.isna().sum().values
    summary['pct_rows_missing'] = df.isna().sum().values / len(df)
    summary.set_index(0, inplace=True)
    summary.index.name=''
    return summary

def null_cols_info(df):
    summary = pd.DataFrame(df.isna().sum(axis=1).values)
    summary.reset_index(inplace=True)
    summary.rename(columns={0:'num_cols_missing'}, inplace=True)
    df2 = summary.groupby('num_cols_missing')\
                                .count().reset_index()
    df2['pct_cols_missing'] = df2.num_cols_missing / df.shape[1]
    df2.rename(columns={'index':'num_rows'}, inplace=True)
    return df2

def handle_missing_values(df, req_column = .7, req_row = .6):
    single_unit = [261, 262, 263, 264, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)] 

    threshold = int(round(req_column*len(df),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    
    threshold = int(round(req_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def extra_clean(df):
    na_cols = df[['parcelid','calculatedbathnbr', 
       'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'fullbathcnt', 'lotsizesquarefeet',
       'regionidcity', 'regionidzip', 'yearbuilt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock']]
    na_cols = pd.DataFrame(KNNImputer().fit_transform(na_cols))
    na_cols.columns = ['parcelid','calculatedbathnbr', 
       'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'fullbathcnt', 'lotsizesquarefeet',
       'regionidcity', 'regionidzip', 'yearbuilt',
       'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
       'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock']
    int_cols = ['parcelid', 'calculatedfinishedsquarefeet',
        'finishedsquarefeet12','fullbathcnt','lotsizesquarefeet',
        'regionidcity','regionidzip', 'yearbuilt',
        'censustractandblock']
    na_cols[int_cols] = na_cols[int_cols].astype(int)
    df = na_cols.merge(df, on='parcelid', copy=False)
    df = df.drop(columns=df.isna().sum()[df.isna().sum()>0]\
        .index.tolist())
    df.drop(columns=['calculatedbathnbr_x', 'propertylandusetypeid',
                 'finishedsquarefeet12_x'], inplace=True)
    df.set_index('parcelid', inplace=True)
    #Drop outliers
    df.drop([11059773], inplace=True)
    df.drop([167655959,167687839,167689317], inplace=True)
    df.drop(df.calculatedfinishedsquarefeet_x.nlargest(3)\
            .index.tolist(), inplace=True)
    df.drop(df.structuretaxvaluedollarcnt_x.nlargest(5).index.tolist(), 
    inplace=True)
    df.drop(df.landtaxvaluedollarcnt_x.nlargest(3).index.tolist(), 
    inplace=True)
    df.drop(df.censustractandblock_x.nlargest(2).index.tolist(), 
    inplace=True)
    df.drop(df.regionidzip_x.nlargest(12).index.tolist(), 
    inplace=True)
    df.drop(df.lotsizesquarefeet_x.nlargest(13).index.tolist(), 
    inplace=True)

    return df
