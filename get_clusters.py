# %%
import warnings
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from engine.models import RFM

# %%
warnings.filterwarnings('ignore')

option_settings = {
    'display.max_columns': None,
    'display.max_rows': False,
    'display.float_format': '{:,.4f}'.format
}
[pd.set_option(setting, option) for setting, option in option_settings.items()]

IN_PATH = 'data/in/'
OUT_PATH = 'data/out/'

#%%
filename = 'raw_data.csv'
data = pd.read_csv(IN_PATH + filename)
subset = data[['order_id', 'fecha_hora', 'document_number', 'precio_full_price_c_promos']].copy()
subset = subset.sample(n=int(round(subset.shape[0]*0.5, 0)), random_state=0)
subset.reset_index(inplace=True)
subset.drop(columns='order_id', inplace=True)
subset.rename(columns={'index': 'order_id'}, inplace=True)
subset['user_id'] = subset['document_number'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

subset.drop(columns='document_number', inplace=True)
new_colnames = {
    'fecha_hora': 'datetime',
    'precio_full_price_c_promos': 'order_value'
}
subset.rename(columns=new_colnames, inplace=True)
subset['order_value'] = subset['order_value'] / 3.9

filename = 'transf_data.csv'
subset.to_csv(OUT_PATH + filename, index=False)

# %%
