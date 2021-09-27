# %%
from operator import sub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('data/2021-09-02 09.csv')
df.iloc[:, 8:]

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
    # plt.show()
    plt.savefig(f'figure/all_{sensor}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)

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
def make_subsequences(df, sensor, window):
    vec = df[sensor].values
    return np.array([vec[i:i+window] for i in range(len(vec)-window+1)])

# %%
for sensor in sensors:
    plt.figure(figsize=(20,4))
    plt.title(sensor)
    subsequences = make_subsequences(df_std, sensor, window)
    plt.plot(subsequences[0])
    # plt.show()
    plt.savefig(f'figure/first_{sensor}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)

# %%
k = 3
kmeans = KMeans(n_clusters=k, random_state=2021)
scaler = StandardScaler()

for sensor in sensors:
    subsequences = make_subsequences(df_std, sensor, window)
    kmeans.fit_predict(subsequences)
    centers = scaler.fit_transform(kmeans.cluster_centers_)
    
    plt.figure(figsize=(20,4))
    plt.title(sensor)
    plt.plot(centers[0], label='cluster1')
    plt.plot(centers[1], label='cluster2')
    plt.plot(centers[2], label='cluster3')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'figure/kmeans_{sensor}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)
# %%
from glob import glob

path_list = glob('data/*')
new_df = pd.DataFrame()

for i, path in enumerate(path_list):
    df = pd.read_csv(path)
    df['ID'] = i
    new_df = pd.concat([new_df, df], axis=0)

new_df = new_df.reset_index(drop=True)
new_df

# %%
new_df_std = scaler(new_df, sensors)
new_df_std

# %%
new_df_std = pd.concat([new_df_std, new_df[['ID']]], axis=1)
new_df_std

# %%
k = 3
kmeans = KMeans(n_clusters=k, random_state=2021)
scaler = StandardScaler()

for sensor in sensors:
    subsequences = np.empty((0, window))
    for i in range(new_df_std['ID'].max()+1):
        one_df = new_df_std[new_df_std['ID']==i]
        one_subsequences = make_subsequences(one_df, sensor, window)
        subsequences = np.append(subsequences, one_subsequences, axis=0)
    print(subsequences.shape)
    kmeans.fit_predict(subsequences)
    centers = scaler.fit_transform(kmeans.cluster_centers_)

    plt.figure(figsize=(20,4))
    plt.title(sensor)
    plt.plot(centers[0], label='cluster1')
    plt.plot(centers[1], label='cluster2')
    plt.plot(centers[2], label='cluster3')
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f'figure/new_kmeans_fixed_{sensor}.png', facecolor="azure", bbox_inches='tight', pad_inches=0)
