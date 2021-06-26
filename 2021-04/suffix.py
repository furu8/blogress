# %%
import pandas as pd
import numpy as np
import glob as gb
from IPython.core.display import display

# %%
def load_df(paths):
    allcity_df = pd.read_csv(paths[0])
    for path in paths[1:]:
        oneyear_df = pd.read_csv(path)
        allcity_df = pd.merge(allcity_df, oneyear_df, on=['年', '月', '日'])
    
    return allcity_df

def load2_df(paths):
    city_names = {'広島', '大阪', '那覇', '高松', '東京'}
    df_dict = make_df_dict(paths, city_names)
    print(df_dict.keys())
    
    allcity_df = df_dict['広島']
    df_dict.pop('広島')
    for city in df_dict.keys():
        allcity_df = pd.merge(allcity_df, df_dict[city], 
                            on=['年', '月', '日'],
                            suffixes=('', f'_{city}'))
    
    return allcity_df

def make_df_dict(paths, city_names):
    df_dict = {}
    for path in paths:
        for city in city_names:
            print(city)
            if city in path:
                df_dict[city] = pd.read_csv(path)
    
    return df_dict

    
def print_columns(df):
    for col in df.columns:
        print(col)

# %%
# データパス
path2016_list = gb.glob('data/2016/*.csv')
path2017_list = gb.glob('data/2017/*.csv')
path2018_list = gb.glob('data/2018/*.csv')
path2019_list = gb.glob('data/2019/*.csv')

# %%
# 本来ならfor文回すが
print(path2016_list)

# %%
# 2016だけ
hiroshima_df = pd.read_csv(path2016_list[0])
osaka_df = pd.read_csv(path2016_list[1])
takamatsu_df = pd.read_csv(path2016_list[2])
tokyo_df = pd.read_csv(path2016_list[3])

display(hiroshima_df.head())
display(osaka_df.head())
display(takamatsu_df.head())
display(tokyo_df.head())

# %%
df = pd.merge(hiroshima_df, osaka_df, on=['年', '月', '日'])
print_columns(df)
# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y

# %%
df = pd.merge(df, takamatsu_df, on=['年', '月', '日'])
print_columns(df)

# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y
# 曜日
# 平均気温
# 最高気温
# 最高気温時間
# 最低気温
# 最低気温時間
# 降水量の合計
# 平均雲量
# 平均風速
# 平均湿度

# %%
df = pd.merge(df, takamatsu_df, on=['年', '月', '日'])
print_columns(df)

# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y

# %%
# 関数定義後
# ロード
df_2016 = load_df(path2016_list)
df_2017 = load_df(path2017_list)
df_2018 = load_df(path2018_list)
df_2019 = load_df(path2019_list)

display(df_2016.head())
display(df_2017.head())
display(df_2018.head())
display(df_2019.head())

# %%
print_columns(df_2016)
# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y

# %%
# 問題1
print(df_2016['平均気温_x'].head())

#    平均気温_x  平均気温_x
# 0     7.6     7.5
# 1     8.1     7.3
# 2    10.9     9.3
# 3    11.2     9.2
# 4    10.0    10.9

# %%
# 問題2
concated_df = pd.concat([df_2016, df_2017], axis=0, ignore_index=True)
print_columns(concated_df)
concated_df.head()

# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y

# %%
concated_df = pd.concat([concated_df, df_2018], axis=0, ignore_index=True)
print_columns(concated_df)
concated_df.head()

# 年
# 月
# 日
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y
# 曜日_x
# 平均気温_x
# 最高気温_x
# 最高気温時間_x
# 最低気温_x
# 最低気温時間_x
# 降水量の合計_x
# 平均雲量_x
# 平均風速_x
# 平均湿度_x
# 曜日_y
# 平均気温_y
# 最高気温_y
# 最高気温時間_y
# 最低気温_y
# 最低気温時間_y
# 降水量の合計_y
# 平均雲量_y
# 平均風速_y
# 平均湿度_y

# %%
# エラー
concated_df = pd.concat([concated_df, df_2019], axis=0, ignore_index=True)

"""
InvalidIndexError: Reindexing only valid with uniquely valued Index objects
"""

# %%
# かぶってなかったらエラーにならない
# https://note.nkmk.me/python-pandas-concat/
df1 = pd.DataFrame({'A': ['A1', 'A2', 'A3'],
                    'B': ['B1', 'B2', 'B3'],
                    'C': ['C1', 'C2', 'C3']})
print(df1)
#       A   B   C
# 0    A1  B1  C1
# 1    A2  B2  C2
# 2    A3  B3  C3

df2 = pd.DataFrame({'A': ['A1', 'A2', 'A3'],
                    'B': ['B1', 'B2', 'B3'],
                    'C': ['C2', 'C3', 'C4'],
                    'D': ['D2', 'D3', 'D4']})
print(df2)
#     A   B   C   D
# 0  A1  B1  C2  D2
# 1  A2  B2  C3  D3
# 2  A3  B3  C4  D4
# %%
df_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df_concat)
#     A   B   C    D
# 0  A1  B1  C1  NaN
# 1  A2  B2  C2  NaN
# 2  A3  B3  C3  NaN
# 3  A1  B1  C2   D2
# 4  A2  B2  C3   D3
# 5  A3  B3  C4   D4

# %%
df3 = pd.DataFrame({'A': ['A1', 'A2', 'A3'],
                    'C': ['C2', 'C3', 'C4'],
                    'D': ['D2', 'D3', 'D4']})

print(df3)
#     A   C   D
# 0  A1  C2  D2
# 1  A2  C3  D3
# 2  A3  C4  D4

# %%
merged_df = pd.merge(df_concat, df1, on='A')
merged_df = pd.merge(merged_df, df2, on='A')
merged_df = pd.merge(merged_df, df3, on='A')
merged_df

"""
C_xとC_yがかぶる
"""

# %%
pd.concat([df1, merged_df])

# %%
# ロード2(改良版)
df2_2016 = load2_df(path2016_list)
df2_2017 = load2_df(path2017_list)
df2_2018 = load2_df(path2018_list)
df2_2019 = load2_df(path2019_list)

display(df2_2016.head())
display(df2_2017.head())
display(df2_2018.head())
display(df2_2019.head())

# %%
print_columns(df2_2016)
# 年
# 月
# 日
# 曜日
# 平均気温
# 最高気温
# 最高気温時間
# 最低気温
# 最低気温時間
# 降水量の合計
# 平均雲量
# 平均風速
# 平均湿度
# 曜日_大阪
# 平均気温_大阪
# 最高気温_大阪
# 最高気温時間_大阪
# 最低気温_大阪
# 最低気温時間_大阪
# 降水量の合計_大阪
# 平均雲量_大阪
# 平均風速_大阪
# 平均湿度_大阪
# 曜日_東京
# 平均気温_東京
# 最高気温_東京
# 最高気温時間_東京
# 最低気温_東京
# 最低気温時間_東京
# 降水量の合計_東京
# 平均雲量_東京
# 平均風速_東京
# 平均湿度_東京
# 曜日_高松
# 平均気温_高松
# 最高気温_高松
# 最高気温時間_高松
# 最低気温_高松
# 最低気温時間_高松
# 降水量の合計_高松
# 平均雲量_高松
# 平均風速_高松
# 平均湿度_高松
# %%
print_columns(df2_2017)
# 年
# 月
# 日
# 曜日
# 平均気温
# 最高気温
# 最高気温時間
# 最低気温
# 最低気温時間
# 降水量の合計
# 平均雲量
# 平均風速
# 平均湿度
# 曜日_大阪
# 平均気温_大阪
# 最高気温_大阪
# 最高気温時間_大阪
# 最低気温_大阪
# 最低気温時間_大阪
# 降水量の合計_大阪
# 平均雲量_大阪
# 平均風速_大阪
# 平均湿度_大阪
# 曜日_東京
# 平均気温_東京
# 最高気温_東京
# 最高気温時間_東京
# 最低気温_東京
# 最低気温時間_東京
# 降水量の合計_東京
# 平均雲量_東京
# 平均風速_東京
# 平均湿度_東京
# 曜日_高松
# 平均気温_高松
# 最高気温_高松
# 最高気温時間_高松
# 最低気温_高松
# 最低気温時間_高松
# 降水量の合計_高松
# 平均雲量_高松
# 平均風速_高松
# 平均湿度_高松

# %%
print_columns(df2_2018)
# 年
# 月
# 日
# 曜日
# 平均気温
# 最高気温
# 最高気温時間
# 最低気温
# 最低気温時間
# 降水量の合計
# 平均雲量
# 平均風速
# 平均湿度
# 曜日_大阪
# 平均気温_大阪
# 最高気温_大阪
# 最高気温時間_大阪
# 最低気温_大阪
# 最低気温時間_大阪
# 降水量の合計_大阪
# 平均雲量_大阪
# 平均風速_大阪
# 平均湿度_大阪
# 曜日_東京
# 平均気温_東京
# 最高気温_東京
# 最高気温時間_東京
# 最低気温_東京
# 最低気温時間_東京
# 降水量の合計_東京
# 平均雲量_東京
# 平均風速_東京
# 平均湿度_東京
# 曜日_高松
# 平均気温_高松
# 最高気温_高松
# 最高気温時間_高松
# 最低気温_高松
# 最低気温時間_高松
# 降水量の合計_高松
# 平均雲量_高松
# 平均風速_高松
# 平均湿度_高松

# %%
print_columns(df2_2019)
# 年
# 月
# 日
# 曜日
# 平均気温
# 最高気温
# 最高気温時間
# 最低気温
# 最低気温時間
# 降水量の合計
# 平均雲量
# 平均風速
# 平均湿度
# 曜日_大阪
# 平均気温_大阪
# 最高気温_大阪
# 最高気温時間_大阪
# 最低気温_大阪
# 最低気温時間_大阪
# 降水量の合計_大阪
# 平均雲量_大阪
# 平均風速_大阪
# 平均湿度_大阪
# 曜日_東京
# 平均気温_東京
# 最高気温_東京
# 最高気温時間_東京
# 最低気温_東京
# 最低気温時間_東京
# 降水量の合計_東京
# 平均雲量_東京
# 平均風速_東京
# 平均湿度_東京
# 曜日_那覇
# 平均気温_那覇
# 最高気温_那覇
# 最高気温時間_那覇
# 最低気温_那覇
# 最低気温時間_那覇
# 降水量の合計_那覇
# 平均雲量_那覇
# 平均風速_那覇
# 平均湿度_那覇
# 曜日_高松
# 平均気温_高松
# 最高気温_高松
# 最高気温時間_高松
# 最低気温_高松
# 最低気温時間_高松
# 降水量の合計_高松
# 平均雲量_高松
# 平均風速_高松
# 平均湿度_高松
# %%
# しいて書くならの後処理
columns = ['曜日', '平均気温', '最高気温', '最高気温時間'
        '最低気温', '最低気温時間', '降水量の合計',
        '平均雲量', '平均風速', '平均湿度']

for col in columns:
    df2_2019 = df2_2019.rename(columns={col: f'{col}_広島'})

# %%
print_columns(df2_2019)
# 年
# 月
# 日
# 曜日_広島
# 平均気温_広島
# 最高気温_広島
# 最高気温時間
# 最低気温
# 最低気温時間_広島
# 降水量の合計_広島
# 平均雲量_広島
# 平均風速_広島
# 平均湿度_広島
# 曜日_大阪
# 平均気温_大阪
# 最高気温_大阪
# 最高気温時間_大阪
# 最低気温_大阪
# 最低気温時間_大阪
# 降水量の合計_大阪
# 平均雲量_大阪
# 平均風速_大阪
# 平均湿度_大阪
# 曜日_東京
# 平均気温_東京
# 最高気温_東京
# 最高気温時間_東京
# 最低気温_東京
# 最低気温時間_東京
# 降水量の合計_東京
# 平均雲量_東京
# 平均風速_東京
# 平均湿度_東京
# 曜日_那覇
# 平均気温_那覇
# 最高気温_那覇
# 最高気温時間_那覇
# 最低気温_那覇
# 最低気温時間_那覇
# 降水量の合計_那覇
# 平均雲量_那覇
# 平均風速_那覇
# 平均湿度_那覇
# 曜日_高松
# 平均気温_高松
# 最高気温_高松
# 最高気温時間_高松
# 最低気温_高松
# 最低気温時間_高松
# 降水量の合計_高松
# 平均雲量_高松
# 平均風速_高松
# 平均湿度_高松