import pandas as pd
import requests
import os

def get_items():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url + '/api/v1/items')
    data = response.json()
    items = pd.DataFrame(data['payload']['items'])
    url = data['payload']['next_page']

    while type(url) != type(None):
        response = requests.get(base_url + url)
        data = response.json()
        items = items.append(data['payload']['items'])
        url = data['payload']['next_page']
    
    return items

def get_stores():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url + '/api/v1/stores')
    data = response.json()
    stores = pd.DataFrame(data['payload']['stores'])
    url = data['payload']['next_page']

    while type(url) != type(None):
        response = requests.get(base_url + url)
        data = response.json()
        stores = stores.append(data['payload']['stores'])
        url = data['payload']['next_page']

    return stores

def get_sales():
    base_url = 'https://python.zach.lol'
    response = requests.get(base_url + '/api/v1/sales')
    data = response.json()
    sales = pd.DataFrame(data['payload']['sales'])
    url = data['payload']['next_page']

    while type(url) != type(None):
        response = requests.get(base_url + url)
        data = response.json()
        sales = sales.append(data['payload']['sales'])
        url = data['payload']['next_page']
    
    sales.to_csv('sales.csv')
    return sales

def combine_all():
    if os.path.isfile('sales.csv'):
        sales = pd.read_csv('sales.csv', index_col=0)
    else:
        sales = get_sales()
    sales = pd.read_csv('sales.csv')
    sales.drop(columns='Unnamed: 0', inplace=True)
    items=get_items()
    stores=get_stores()
    total_data = pd.merge(sales, items, left_on='item', 
                        right_on='item_id', how='left')
    total_data = pd.merge(total_data, stores, left_on='store', 
                        right_on='store_id', how='left')
    total_data.drop(columns=['item','store'], inplace=True)
    total_data.to_csv('total_data.csv')
    return total_data

def get_power_data():
    df = pd.read_csv('https://raw.githubusercontent.com/jenfly/\
opsd/master/opsd_germany_daily.csv')
    return df