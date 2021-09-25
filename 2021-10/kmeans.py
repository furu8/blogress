# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('2021-09-02 09.csv')
df

# # %%
# df['dif_sec'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S').diff().dt.total_seconds().fillna(0)
# df['cum_sec'] = df['dif_sec'].cumsum()
# df

# %%
sensors = [
    'Acc_X', 'Acc_Y', 'Acc_Z',
    'Roll', 'Pitch', 'Yaw',
    'Com_X', 'Com_Y', 'Com_Z'
]
# %%
for sensor in sensors:
    plt.figure(figsize=(20,4))
    plt.title(sensor)
    plt.plot(df[sensor])
    plt.show()

# %%
def scaler(df, sensors):
    scaler = StandardScaler()
    arr = scaler.fit_transform(df[sensors])
    df_std = pd.DataFrame(arr, columns=sensors)
    return df_std

# %%
df_std = scaler(df, sensors)
df_std
# %%
window = 128
def make_subsequences(df, sensor):
    vec = df[sensor].values
    return np.array([vec[i:i+window] for i in range(len(vec)-window+1)])

# %%
for sensor in sensors:
    plt.figure(figsize=(20,4))
    plt.title(sensor)
    subsequences = make_subsequences(df_std, sensor)
    plt.plot(subsequences[0])
    plt.show()

# %%
kmeans = KMeans(n_clusters=3, random_state=2021)
scaler = StandardScaler()

for sensor in sensors:
    subsequences = make_subsequences(df_std, sensor)
    kmeans.fit_predict(subsequences)
    centers = scaler.fit_transform(kmeans.cluster_centers_)

    plt.figure(figsize=(20,4))
    plt.title(sensor)
    plt.plot(centers[0], label='cluster1')
    plt.plot(centers[1], label='cluster2')
    plt.plot(centers[2], label='cluster3')
    plt.legend(loc='upper right')
    plt.show()
