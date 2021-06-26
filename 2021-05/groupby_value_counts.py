# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# %%
df = pd.DataFrame([
        ['Aさん', 100, 'S', 'cola'],
        ['Bさん', 150, 'M', 'tea'],
        ['Cさん', 200, 'L', 'tea'],
        ['Dさん', 100, 'S', 'tea'],
        ['Eさん', 200, 'L', 'coffee'],
        ['Fさん', 200, 'L', 'tea'],
        ['Gさん', 150, 'M', 'coffee'],
        ['Hさん', 200, 'L', 'coffee'],
        ['Iさん', 100, 'S', 'cola'],
        ['Jさん', 200, 'L', 'tea'],
        ['Kさん', 200, 'L', 'tea'],
        ['Lさん', 100, 'S', 'cola'],
        ['Mさん', 150, 'M', 'coffee'],
        ['Nさん', 150, 'M', 'coffee'],
        ],
        columns=['user', 'price', 'size', 'drink'])
print(df)

#    user  price size   drink
# 0   Aさん    100    S    cola
# 1   Bさん    150    M     tea
# 2   Cさん    200    L     tea
# 3   Dさん    100    S     tea
# 4   Eさん    200    L  coffee
# 5   Fさん    200    L     tea
# 6   Gさん    150    M  coffee
# 7   Hさん    200    L  coffee
# 8   Iさん    100    S    cola
# 9   Jさん    200    L     tea
# 10  Kさん    200    L     tea
# 11  Lさん    100    S    cola
# 12  Mさん    150    M  coffee
# 13  Nさん    150    M  coffee

# %%
# ラベリング
lenc = LabelEncoder()
df['drink'] = lenc.fit_transform(df['drink']) # 0=coffe, 1=cola, 2=tea
print(df)

#    user  price size  drink
# 0   Aさん    100    S      1
# 1   Bさん    150    M      2
# 2   Cさん    200    L      2
# 3   Dさん    100    S      2
# 4   Eさん    200    L      0
# 5   Fさん    200    L      2
# 6   Gさん    150    M      0
# 7   Hさん    200    L      0
# 8   Iさん    100    S      1
# 9   Jさん    200    L      2
# 10  Kさん    200    L      2
# 11  Lさん    100    S      1
# 12  Mさん    150    M      0
# 13  Nさん    150    M      0
# %%
df['drink'].value_counts()

df['drink'].value_counts()
# 2    6
# 0    5
# 1    3
# Name: drink, dtype: int64
# %%
# 0=coffe, 1=cola, 2=tea
df['drink'].value_counts().idxmax() 
# %%
df.groupby('size').agg({'drink':'value_counts'})
# %%
print(df.groupby('size').mean()[['drink']])

#          drink
# size          
# L     1.333333
# M     0.500000
# S     1.250000
# %%
print(pd.DataFrame(df.groupby('size').apply(lambda df: df['drink'].value_counts().idxmax()), columns=['drink']))

#       drink
# size       
# L         2
# M         0
# S         1
