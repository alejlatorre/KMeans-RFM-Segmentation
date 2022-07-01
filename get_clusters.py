# %% Libraries
import warnings
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from yellowbrick.cluster import silhouette_visualizer

from engine.models import RFM

# Settings
warnings.filterwarnings('ignore')

option_settings = {
    'display.max_columns': None,
    'display.max_rows': False,
    'display.float_format': '{:,.4f}'.format
}
[pd.set_option(setting, option) for setting, option in option_settings.items()]

IN_PATH = 'data/in/'
OUT_PATH = 'data/out/'

# Import data
dtypes_dict = {
    'order_id': 'str',
    'datetime': 'str',
    'user_id': 'str'
}
filename = 'raw_data.csv'
data = pd.read_csv(IN_PATH + filename, dtype=dtypes_dict)
data.head()

# %% Get RFM variables and scores
model = RFM()
rfm_vars = model.get_vars(data=data, user_col='user_id', date_col='datetime', order_col='order_id', ticket_col='order_value')
rfm_scores = model.get_scores(data=rfm_vars, r_col='recency', f_col='frequency', m_col='monetary')

# %% Standardize
raw_cols = ['frequency', 'monetary', 'recency']
X = rfm_vars[raw_cols].copy()

scaler = MinMaxScaler()
norm_cols = ['norm_frequency', 'norm_monetary', 'norm_recency']
rfm_vars[norm_cols] = scaler.fit_transform(X)
rfm_vars.describe()

# %% Get optimal k-values
WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i, init = 'k-means++')
    model.fit(rfm_vars[norm_cols])
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (8, 5))
plt.plot(range(1,11), WCSS, linewidth=4, markersize=7, marker='o', color = 'green')
plt.xticks(np.arange(11))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.title('Elbow Method')
plt.show()

# %% Silhouette score for k-values
print('Silhouette score for:')
for i in range(3, 11):
    labels=KMeans(n_clusters=i, init='k-means++', random_state=0).fit(X_t).labels_
    score=silhouette_score(X_t, labels, metric='euclidean', random_state=0)
    print(f'{i} clusters: {score}')

silhouette_visualizer(KMeans(n_clusters=3, random_state=0), X_t, colors='yellowbrick')
plt.show()

# %% KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(rfm_vars[norm_cols])
rfm_vars['cluster'] = kmeans.predict(rfm_vars[norm_cols])

# Boxplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
sns.boxplot(x=rfm_vars['cluster'], y=rfm_vars['recency'], ax=axes[0])
sns.boxplot(x=rfm_vars['cluster'], y=rfm_vars['frequency'], ax=axes[1])
sns.boxplot(x=rfm_vars['cluster'], y=rfm_vars['monetary'], ax=axes[2])
plt.suptitle('Variables per cluster')
plt.show()

# %% Mapping values
new_values = {
    2: '1. High',
    0: '2. Mid',
    1: '3. Low'
}
rfm_vars['_cluster'] = rfm_vars['cluster'].map(new_values)

# Boxplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 7))
sns.boxplot(x=rfm_vars['_cluster'], y=rfm_vars['recency'], ax=axes[0])
sns.boxplot(x=rfm_vars['_cluster'], y=rfm_vars['frequency'], ax=axes[1])
sns.boxplot(x=rfm_vars['_cluster'], y=rfm_vars['monetary'], ax=axes[2])
plt.suptitle('Variables per cluster')
plt.show()


# %%
