# はじめに

複数ファイルにわかれたデータの結合方法のメモです。  

**InvalidIndexError: Reindexing only valid with uniquely valued Index objects**  
というエラーとも戦いました。

<br>
#  データ

[気象庁の気象データ](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)を使いました。  
広島、高松、大阪、東京、那覇の5都市分のデータです。  

Pythonスクリプトと同じ階層にあるdataディレクトリの中身は以下になります。

```
├── data                      
│   ├── 2016               <- 2016年の気象データ
│   │   ├── 広島.csv
│   │   ├── 高松.csv
│   │   ├── 大阪.csv
│   │   ├── 東京.csv
│   ├── 2017               <- 2017年の気象データ
│   │   ├── 広島.csv
│   │   ├── 高松.csv
│   │   ├── 大阪.csv
│   │   ├── 東京.csv
│   ├── 2018               <- 2018年の気象データ
│   │   ├── 広島.csv
│   │   ├── 高松.csv
│   │   ├── 大阪.csv
│   │   ├── 東京.csv
│   ├── 2019               <- 2019年の気象データ
│   │   ├── 広島.csv
│   │   ├── 高松.csv
│   │   ├── 大阪.csv
│   │   ├── 東京.csv
│   │   ├── 那覇.csv
```

<br>
dataを表にするとこんな感じです。

<figure class="figure-image figure-image-fotolife" title="data">[f:id:Noleff:20210626192743j:plain]<figcaption>data</figcaption></figure>

<br>
# 結合方法

以下の1と2の方法があるかと思います。  
今回はどちらが良いかの議論はしませんが、ケースバイケースな気がします。  
1で本記事は進めます。

1. merge→concat

    <figure class="figure-image figure-image-fotolife" title="merge→concat">[f:id:Noleff:20210626192810j:plain]<figcaption>merge→concat</figcaption></figure>

2. concat→merge

    <figure class="figure-image figure-image-fotolife" title="concat→merge">[f:id:Noleff:20210626192843j:plain]<figcaption>concat→merge</figcaption></figure>

<br>
# 結合手順

#### 手始めに

まず、データをロードするために、ファイルのパスを取得します。


```python
# %%
import pandas as pd
import numpy as np
import glob as gb
from IPython.core.display import display

# %%
# カラムを表示
def print_columns(df):
    for col in df.columns:
        print(col)

# %%
# データパス
path2016_list = gb.glob('data/2016/*.csv')
path2017_list = gb.glob('data/2017/*.csv')
path2018_list = gb.glob('data/2018/*.csv')
path2019_list = gb.glob('data/2019/*.csv')
```

<br>
2016年だけとりあえず結合してみます。

<figure class="figure-image figure-image-fotolife" title="merge">[f:id:Noleff:20210626193001j:plain]<figcaption>merge</figcaption></figure>

```python
# %%
# 雑に2016だけ
print(path2016_list)
# ['data/2016\\大阪.csv', 'data/2016\\広島.csv', 'data/2016\\東京.csv', 'data/2016\\高松.csv']

hiroshima_df = pd.read_csv(path2016_list[0])
osaka_df = pd.read_csv(path2016_list[1])
takamatsu_df = pd.read_csv(path2016_list[2])
tokyo_df = pd.read_csv(path2016_list[3])

display(hiroshima_df.head())
display(osaka_df.head())
display(takamatsu_df.head())
display(tokyo_df.head())
```

<figure class="figure-image figure-image-fotolife" title="各都市の気象データ">[f:id:Noleff:20210626181154p:plain]<figcaption>各都市の気象データ</figcaption></figure>

<br>
3回結合します。  
このとき、**カラムが重複するという問題**が発生します。

```python
# %%
# merge①
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

# merge②
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

# merge③
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
```

③で重複が発生しているのがわかります。

<br>
#### そもそも、\_x、\_yは何なのか

これはpd.mergeメソッドの引数suffixesのデフォルト値です。  
重複すると_xと_yとつけるようになっています。  
しかし、同じカラムのあるデータを3回結合すると、重複は再び起きます。  

**これを回避するのが本記事の目的です。**

<br>
#### 各都市でやる

```python
# %%
def load_df(paths):
    allcity_df = pd.read_csv(paths[0])
    for path in paths[1:]:
        oneyear_df = pd.read_csv(path)
        allcity_df = pd.merge(allcity_df, oneyear_df, on=['年', '月', '日'])
    
    return allcity_df

# %%
# ロード
df_2016 = load_df(path2016_list)
df_2017 = load_df(path2017_list)
df_2018 = load_df(path2018_list)
df_2019 = load_df(path2019_list)
```

<br>
もちろん、同じ問題が起きます。

```python
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
```

<br>
#### 問題点

カラムが重複すると何が問題でしょうか。  
今回は二つの例で示します。

<br>
1. 一度に二つのカラムが触れてしまう

単に出力しても、計算しても、代入しても全部二つのカラムを指定してしまいます。

```python
df_2016['平均気温_x'].head()

#    平均気温_x  平均気温_x
# 0       7.6       7.5
# 1       8.1       7.3
# 2      10.9       9.3
# 3      11.2       9.2
# 4      10.0      10.9
```

<br>
2. concatするとエラー

merge→concatで結合する今回は、mergeした各都市のデータを年ごとにconcatしなければなりません。  
そのとき、カラムが重複しているエラーが出ます。  
それが**InvalidIndexError: Reindexing only valid with uniquely valued Index objects**です。

<figure class="figure-image figure-image-fotolife" title="merge→concat">[f:id:Noleff:20210626192810j:plain]<figcaption>【再掲】merge→concat</figcaption></figure>

<br>
concat①、concat②の時点でもカラムは重複していますが、カラムが完全一致していたため、エラーが出ることなく結合できていたようです。  
しかし、2019年だけに那覇の気象データがあるような（データがある時から増えるようなケースが往々にしてあるという前提）場合において、カラムが完全一致しないため、上記のエラーが出るようです。

```python
# %%
# concat① (2016と2017の結合)
concated_df = pd.concat([df_2016, df_2017], axis=0, ignore_index=True)
print_columns(concated_df)

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
# concat② (上記dfと2018の結合)
concated_df = pd.concat([concated_df, df_2018], axis=0, ignore_index=True)
print_columns(concated_df)
# concat① (2016と2017の結合)と同じ結果


# %%
# concat③ (上記dfと2019の結合)
# エラー
concated_df = pd.concat([concated_df, df_2019], axis=0, ignore_index=True)
# InvalidIndexError: Reindexing only valid with uniquely valued Index objects
```

<br>
#### 重複していない場合

簡単なデータで重複していないときのconcatの挙動を見てみます。  
[こちら](https://note.nkmk.me/python-pandas-concat/)を参考にコードを作成しました。  
データがない箇所はNaNになり、問題なく結合自体はできていることがわかるかと思います。

```python
# %%
# かぶってなかったらエラーにならない
df1 = pd.DataFrame({'A': ['A1', 'A2', 'A3'],
                    'B': ['B1', 'B2', 'B3'],
                    'C': ['C1', 'C2', 'C3']})
print(df1)
#       A   B   C
# 0    A1  B1  C1
# 1    A2  B2  C2
# 2    A3  B3  C3

# %%
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
```

<br>
#### 改良版の作成

先程のload_df関数を改良します。    
ただ、問題はこの改良方法がコード的に美しくない（少々強引）ことです。    
他に良い案があれば、是が非でもご教示していただきたく……。

**アイデアはデータを辞書に入れることと、その辞書のkeyを引数suffixesに指定することです。**

```python
# %%
# load_dfを改良
def load2_df(paths):
    city_names = {'広島', '大阪', '那覇', '高松', '東京'}
    df_dict = make_df_dict(paths, city_names)
    
    allcity_df = df_dict['広島']
    df_dict.pop('広島')
    for city in df_dict.keys():
        allcity_df = pd.merge(allcity_df, df_dict[city], 
                            on=['年', '月', '日'],
                            suffixes=('', f'_{city}'))
    
    return allcity_df

# 各都市のデータを辞書に入れる
def make_df_dict(paths, city_names):
    df_dict = {}
    for path in paths:
        for city in city_names:
            if city in path:
                df_dict[city] = pd.read_csv(path)
    
    return df_dict

# %%
# ロード2(改良版)
df2_2016 = load2_df(path2016_list)
df2_2017 = load2_df(path2017_list)
df2_2018 = load2_df(path2018_list)
df2_2019 = load2_df(path2019_list)

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
# df2_2016と同じ

# %%
print_columns(df2_2018)
# df2_2016と同じ

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
```

<br>
2019年にしかない那覇も問題なくできました。

<br>
#### 広島は？

\_都市名となっていないデータが広島であることは、コードを書いた本人（私）しかわかりません。  
これも強引に後処理を書くとこんな感じでしょうか。

```python
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
```

<br>
# おわりに

複数ファイルのデータ結合方法について直面した問題をまとめました。  
もっとスマートに書けないもんか。

<br>
# 参考文献

[https://www.data.jma.go.jp/gmd/risk/obsdl/:title]

[https://note.nkmk.me/python-pandas-concat/:title]

