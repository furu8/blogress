# %%
import pandas as pd

# %%
def add_digits(df, col):
    df[col] = df[col].astype(str)
    df['count'] = df[col].apply(lambda x: len(x))
    df.loc[df['count']==1, col] = '0' + df[col]
    df = df.drop('count', axis=1)

    return df


# %%
# ロード
df = pd.read_csv('data/data_naha.csv')
df

# %%
# 2桁に変換
df = add_digits(df, '最高気温時')
df = add_digits(df, '最高気温分')
df = add_digits(df, '最低気温時')
df = add_digits(df, '最低気温分')
df.head(50)

# %%
# 時と分を結合
df['最高気温時間'] = df['最高気温時'] + ':' + df['最高気温分']
df['最低気温時間'] = df['最低気温時'] + ':' + df['最低気温分']
df

# %%
# 除去and並び替え
df = df.drop(['最高気温時', '最高気温分', '最低気温時', '最低気温分'], axis=1)
df = df.reindex(columns=['年', '月', '日', '曜日', '平均気温', '最高気温', '最高気温時間', '最低気温', '最低気温時間', '降水量の合計', '平均雲量', '平均風速', '平均湿度'])
df
# %%
# セーブ
df.to_csv('data/weather_naha.csv', index=False)
# %%
