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
plt.figure(figsize=(20,4))
plt.plot(df['Acc_X'])
plt.show()

# %%
scaler = StandardScaler()
train = scaler.fit_transform(df[['Acc_X']])
train = train.flatten()
train

# %%
plt.figure(figsize=(20,4))
plt.plot(train)
plt.show()

# %%
window = 128
subsequences = np.array([train[i:i+window] for i in range(0,len(train)-window+1,3)])
subsequences.shape

# %%
plt.plot(subsequences[0])
plt.show()

# %%
kmeans = KMeans(n_clusters=3, random_state=2021)
centers = []

for i, subseq in enumerate(subsequences):
    kmeans.fit_predict(subseq.reshape(-1,1))
    centers.append(kmeans.cluster_centers_.flatten())
    print(i)

centers = np.array(centers)
centers.shape

# %%
plt.figure(figsize=(20,4))
for i, center in enumerate(centers):
    print(i)
    plt.scatter([i, i+1, i+2], center, color='tab:blue')
plt.plot(np.sin(range(len(centers))), color='black')
plt.show()

# %%
plt.figure(figsize=(20,4))
plt.scatter([0,1,2], centers[0])
plt.scatter([3,4,5], centers[1])
plt.scatter([6,7,8], centers[2])
plt.scatter([9,10,11], centers[3])
plt.plot(np.sin(range(12)))
plt.show()
# %%
pred = kmeans.fit_predict(train)
pred
# %%
kmeans.cluster_centers_ 

# %%
plt.figure(figsize=(20,4))
plt.plot(train[:,0], train[:,1], alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_ [:,1], color='r')
plt.show()
# %%
