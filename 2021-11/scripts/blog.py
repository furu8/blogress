# %%
import pandas as pd
import numpy as np
import glob as gb
import re

# 最大表示行数を設定
pd.set_option('display.max_rows', 1000)
# 最大表示列数の指定
pd.set_option('display.max_columns', 1000)

# %%
flowers_labels = [re.split('[\\\.]',path)[-1] for path in gb.glob('D:/OpenData/flowers/raw/*')]
flowers_labels

# %%
fruit_vegetable_labels = [re.split('[\\\.]',path)[-1] for path in gb.glob('D:/OpenData/Fruit-and-Vegetable-Image-Recognition/train/*')]
fruit_vegetable_labels

# %%
labels = flowers_labels + fruit_vegetable_labels
labels_df = pd.DataFrame()
labels_df['labels'] = labels
labels_df = labels_df.reset_index()
labels_df['index'] = labels_df['index']
labels_df
# %%
df = pd.read_csv('../data/result.csv')
df
# %%
df = pd.merge(df, labels_df, left_on='pred_base', right_on='index', how='left').drop(columns='index').rename(columns={'labels':'base_pokemon'})
df = pd.merge(df, labels_df, left_on='pred_inception', right_on='index', how='left').drop(columns='index').rename(columns={'labels':'inception_pokemon'})
df

# %%
sinnoh_df = pd.read_csv('../data/sinnoh_pokemon.csv')
sinnoh_df
# %%
df = pd.merge(df, sinnoh_df, left_on='sinnoh_pokemon', right_on='number').drop(columns='sinnoh_pokemon')
df
# %%
df.drop(['pred_base', 'pred_inception', 'number'], axis=1)
# %%
