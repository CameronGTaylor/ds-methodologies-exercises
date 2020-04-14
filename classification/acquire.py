import pandas as pd
import numpy as np
import env
from pydataset import data

def get_iris_data():
    url = env.get_db_url('iris_db')
    query = '''
    SELECT * FROM measurements
    JOIN species USING (species_id)
    '''
    return pd.read_sql(query, url)
    
def get_titanic_data():
    url = env.get_db_url('titanic_db')
    query = '''
    SELECT * FROM passengers
    '''
    return pd.read_sql(query, url)