from env import get_db_url
import pandas as pd
import numpy as np


def wrangle_telco():
    url = get_db_url('telco_churn')
    query = ('''
        SELECT customer_id, monthly_charges, tenure, total_charges
        FROM customers
        WHERE contract_type_id = 3
    ''')
    df = pd.read_sql(query, url)

    df.total_charges = df.total_charges.str.strip()
    df.total_charges = df.total_charges.replace('', 0).astype(float)
    return df
