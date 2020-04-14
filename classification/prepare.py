import pandas as pd
import numpy as np
import env, acquire, split_scale
from pydataset import data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

def prep_iris(df):
    df.drop(columns=['species_id','measurement_id'], inplace=True)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 
                    'petal_width', 'species']

    train, test = split_scale.split_my_data(df, .8)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(train[['species']])
    cols = [c for c in encoder.categories_[0]]
    m_train = encoder.transform(train[['species']])
    m_test = encoder.transform(test[['species']])
    encoded_train = pd.DataFrame(m_train, columns=cols, 
                                index=train.index)
    encoded_test = pd.DataFrame(m_test, columns=cols, 
                                index=test.index)
    train = pd.concat([train, encoded_train], axis=1).drop(
        columns='species')
    test = pd.concat([test, encoded_test], axis=1).drop(
        columns='species')
    
    return train, test

def prep_titanic(df):
    df.drop(columns=['deck'], inplace=True)
    df.embark_town = df.embark_town.fillna('Southampton')
    df.embarked = df.embarked.fillna('S')

    train, test = split_scale.split_my_data(df, .8)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(train[['embarked']])
    cols = [c for c in encoder.categories_[0]]
    m_train = encoder.transform(train[['embarked']])
    m_test = encoder.transform(test[['embarked']])

    encoded_train = pd.DataFrame(m_train, columns=cols, 
                                index=train.index)
    encoded_test = pd.DataFrame(m_test, columns=cols, 
                                index=test.index)

    train = pd.concat([train, encoded_train], axis=1).drop(
        columns='embarked')
    test = pd.concat([test, encoded_test], axis=1).drop(
        columns='embarked')

    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train[['age']])
    train.age = imputer.transform(train[['age']])
    test.age = imputer.transform(test[['age']])
    
    train_to_scale = train[['age','fare']]
    test_to_scale = test[['age','fare']]
    scaler, train_scaled, test_scaled = \
        split_scale.min_max_scaler(train_to_scale, test_to_scale)

    train.update(train_scaled)
    test.update(test_scaled)

    

    return train, test