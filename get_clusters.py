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


# %%
