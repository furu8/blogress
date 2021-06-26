# %%
import pandas as pd
import os

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
# 年毎に抽出し保存
def save_extracted_year(df, year_list, city='tokyo'):
    for year in year_list:
        oneyear_df = df[df['年']==year].reset_index(drop=True)
        if not os.path.exists(f'data/{year}'):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(f'data/{year}')
        oneyear_df.to_csv(f'data/{year}/weather_{city}.csv', index=False)


# %%
year_list = [2016, 2017, 2018, 2019]
save_extracted_year(tokyo_df, year_list, city='tokyo')
save_extracted_year(osaka_df, year_list, city='osaka')
save_extracted_year(takamatsu_df, year_list, city='takamatsu')
save_extracted_year(hiroshima_df, year_list, city='hiroshima')