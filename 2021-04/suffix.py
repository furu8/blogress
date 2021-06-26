# %%
import pandas as pd

# %%
# データ
tokyo_df = pd.read_csv('data/weather_tokyo.csv')
osaka_df = pd.read_csv('data/weather_osaka.csv')
takamatsu_df = pd.read_csv('data/weather_takamatsu.csv')
hiroshima_df = pd.read_csv('data/weather_hiroshima.csv')

display(tokyo_df.head())
display(osaka_df.head())
display(takamatsu_df.head())
display(hiroshima_df.head())

# %%
adf = pd.merge(tokyo_df, osaka_df, on=['年', '月', '日'])
for col in df.columns:
    print(col)


# %%
adf = pd.merge(df, takamatsu_df, on=['年', '月', '日'])
for col in df.columns:
    print(col)


# %%
adf = pd.merge(df, hiroshima_df, on=['年', '月', '日'])
for col in df.columns:
    print(col)


# %%
pd.concat([df, adf])