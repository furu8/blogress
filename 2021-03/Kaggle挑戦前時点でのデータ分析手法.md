# はじめに

最近、重い腰を上げ、ようやくKaggleを始めました。  
タイタニックやインターン限定のコンペ等には参加したことがありましたが、賞金が発生するようなKaggleに参加したことは、今までありませんでした。

データサイエンス及びエンジニアリングのスキルは研究メインで勉強している現状です。  
そんな自分が、現時点でデータを与えられた場合、何から初めてどう進めるかのプロセスを本記事でまとめたいと思います。

Kaggle等進め、自分にさらに技術力がついたとき、この記事を読んで「このときはわかってなかった……」と顧みるための備忘録とも言えます。  
なお、ビジネス的な話はなしとします。

<br>
####  データ

データはタイタニックのデータを使います。  
あくまで、どういう手順で分析を進めるかに重きを置くので、精度や特徴ベクトルの有用性などは検討しません。

タイタニックの各カラムがどんなデータかは、以下を参照してください。

[https://www.kaggle.com/c/titanic/data:embed:cite]


<br>
# フォルダ構成

フォルダ構成は以下になります。  
[こちら](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)を参考にしました。

各スクリプトとdataフォルダ内の関係性は次節以降で述べます。

```
├── data                      <- データ関連
│   ├── interim               <- 作成途中のデータ
│   ├── processed             <- 学習に使うデータ
│   ├── raw                   <- 生データ
│   │   ├── test.csv
│   │   ├── train.csv
│   ├── submission            <- 提出用データ
│   │   ├── sample_submission.csv
├── scripts                   <- プログラム類
│   ├── models                <- モデル類
│   │   ├──__init__.py
│   │   ├── model.py          <- モデル基底クラス
│   │   ├── model_lgb.py      <- LightGBMクラス
│   │   ├── util.py           <- 汎用クラス
│   ├── generate.py           <- データ作成
│   ├── analyze.py            <- 分析用スクリプト
│   ├── run.py                <- 学習用スクリプト
│   ├── config                <- 汎用的処理クラス
│   │   ├── features          <- 特徴量のカラム群
├── models                    <- 作成したモデル保存
```

<br>
# 手順

1. データを作る
2. データを分析する
3. モデルを作り、評価する

<br>
#### 1. データを作る

ここでの処理では、例えば、複数ファイルに別れたファイルを分析しやすいように一つのファイルにまとめるなどがあります。  
タイタニックはもちろん比較的きれいなデータなので、そうはなっていません。  
一見すると必要のない処理かもしれませんが、ここでは学習データとテストデータを分析するために、2つのデータを結合します。

まず、データセットを読み込み、中身を見ます。

```python
import os
import pandas as pd
from IPython.core.display import display

## データセット作成
# 読込
train_df = pd.read_csv('../data/raw/train.csv')
test_df = pd.read_csv('../data/raw/test.csv')

display(train_df.head())
display(test_df.head())
```

<br>
欠損と基本的な統計値の確認をします。

```python
# 欠損確認
display(train_df.info())
display(test_df.info())

# 統計値確認
display(train_df.describe())
display(test_df.describe())
```

<br>
以下は、# 欠損確認 の出力ですが、学習もテストデータもAgeとCabinが大きく欠損していることがわかります。
```
# 欠損確認...
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
```

<br>
学習データとテストデータを結合します。  
結合するメリットはいくつかありますが、結合することで、一括にデータ補完やラベリングできることが主な理由だと思っています。  
なお、テストデータには目的変数Survivedがないため、-1を補完しています。

```python
# 結合
df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)
df['Survived'] = df['Survived'].fillna(-1)

display(df)
display(df.info())
display(df.describe())
```

<br>
欠損が確認された、Age、Embarked、Fareのデータを補完します。  
補完の方法はいくつかありますが、今回は以下のようにしました。

- Age：年齢の平均で補完
- Embarked：最も多かったカテゴリ（港）で補完
- Fare：料金の平均で補完

```python
# 欠損補完
df['Age'] = df['Age'].fillna(df['Age'].mean()) # 29.881137667304014
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].value_counts().idxmax()) # S
df['Fare'] = df['Fare'].fillna(df['Fare'].mean()) # 33.295479281345564
``` 

<br>
不必要なカラムを削除します。  
本来ならば、あるカラムのデータが必要化どうかは、可視化して分析したり、モデルに入れてみて効くか見たりなどしてから、不要かどうか判断すべきです。  
もし、カラムが不要であることが自明、もしくはあらかじめわかっていた場合、ここで削除する処理を入れるようにしています。

```python
# カラム削除
df = df.drop(['Name', 'Ticket'], axis=1)
```

<br>
学習データとテストデータを見分けるタグ用のカラムを用意します。  
Survivedの値からわかることではありますが、今回は、ブログ用にわかりやすくするため入れました。

```python
# trainとtest
df.loc[df['Survived']!=-1, 'data'] = 'train'
df.loc[df['Survived']==-1, 'data'] = 'test'

display(df)
display(df.info())
display(df.describe())
```

<br>
最後に保存します。
生データから新たに生成したデータなため、中間データとしてinterimフォルダに保存しています。

```python
# 保存
df.to_csv('../data/interim/all.csv', index=False)
```

<br>
#### 2. データを分析する

ここでの処理では、機械学習の肝とも呼べる前処理をとにかく、しまくります。  
実際にする大まか処理は以下になります。

- 生データの特徴量（カラム）の可視化
- 新たな特徴量作成
- 主成分分析
- クラスタリング
- 新たに作成した特徴量の可視化

今回は、可視化をメインに新たな特徴量作成は必要最低限としました。

```python
import os
import pandas as pd
import numpy as np
from IPython.core.display import display
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from models import Util 

# 読込
df = pd.read_csv('../data/interim/all.csv')
df.head()
```

<br>
生データの特徴量（カラム）の可視化をします。  
まずは、死亡者と生存者に違いが見られる特徴量を探してみることにします。  
Pclass、Sex、SibSp、Parch、Embarkedについて可視化します。

```python
# 死亡者と生存者の違い
df_s = df[df['data']=='train']
cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df_s, hue=df_s['Survived'])
    plt.legend( loc='upper right')
    plt.show()
```

<br>
下図の結果から、以下のことがわかります。

- Pclass：3はチケットのクラスがLowerため、多く亡くなっている
- Sex：女性より男性の方が多く亡くなっている
- SibSpとParch：自分以外に一人か二人、家族等がいた方が生存している
- Cherbourg港だけ、生存者が多い

|  特徴量  | 棒グラフ  | 
| ---- | ---- |
| Pclass | [f:id:Noleff:20210531102733p:plain:w200:h150] | 
| Sex | [f:id:Noleff:20210531102827p:plain:w200:h150] | 
| SibSp | [f:id:Noleff:20210531102846p:plain:w200:h150] |
| Parch | [f:id:Noleff:20210531102908p:plain:w200:h150] | 
| Embarkd | [f:id:Noleff:20210531102933p:plain:w200:h150]

<br>
新たに特徴量を作成します。  
カラムSibSpとParchから、家族の人数という新しい特徴量を作ります。  
これは、SibSp – タイタニックに同乗している兄弟/配偶者の数、parch – タイタニックに同乗している親/子供の数という2つの特徴を1つにまとめる処理ともいえます。  

```python
# 家族人数
df['Family'] = df['SibSp'] + df['Parch']
```

<br>
また、機械学習するにはカテゴリ変数を数値に変える必要があります。   
これは、以前[こちら](https://noleff.hatenablog.com/entry/2021/01/03/013245)の記事でも書きました。よければ参考にしてください。  
Sex、Embarked、Cabinに関して、ラベルエンコーディングをします。  
なお、Cabinのみ文字の頭だけ抽出する処理をしています。

```python
# ラベルエンコーディング
lenc = LabelEncoder()

lenc.fit(df['Sex'])
df['Sex'] = pd.DataFrame(lenc.transform(df['Sex']))

lenc.fit(df['Embarked'])
df['Embarked'] = pd.DataFrame(lenc.transform(df['Embarked']))

df['Cabin'] = df['Cabin'].apply(lambda x:str(x)[0])
lenc.fit(df['Cabin'])
df['Cabin'] = pd.DataFrame(lenc.transform(df['Cabin']))
```

<br>
作成した新たな特徴量を可視化します。  
コードはcols以外変わりません（関数にしていないのはご愛嬌ということで）。  
なお、SexとEmbarkedは、先ほど可視化しているので省略しています。

```python
# 死亡者と生存者の違い
df_s = df[df['data']=='train']
cols = ['Family', 'Cabin']
for col in cols:
    print(col)
    plt.figure(figsize=(4,3))
    sns.countplot(x=col, data=df_s, hue=df_s['Survived'])
    plt.legend( loc='upper right')
    plt.show()
```

<br>
下図の結果から、以下のことがわかります。

- Family：家族が1人～3人いる人は生存している可能性が高い
- Cabin：欠損値になっている人がかなり亡くなっている。また、1~5（B～F）は生存者の方が多い

|  特徴量  | 棒グラフ  | 
| ---- | ---- |
| Family | [f:id:Noleff:20210531105340p:plain:w200:h150] | 
| Cabin | [f:id:Noleff:20210531105443p:plain:w200:h150] | 

<br>
作られたデータフレームを保存します。  
中間データから作成したものの内、そのまま学習データになるものはprocessedフォルダに保存しています。  
中間データから中間データを作成することも当然あり、その場合は別の中間データとしてinterimフォルダに保存します。  
今回は前者です。

```python
# 保存
df.to_csv('../data/interim/all.csv', index=False)
```

<br>
最後に特徴量をダンプしておきます。  
util.py含め、modelsフォルダ内のコードはAppendixを見てください。

```python
# 特徴量
df = df.drop(['PassengerId', 'data', 'Survived'], axis=1)

# 特徴量保存
Util.dump(df.columns, 'config/features/all.pkl')
```

<br>
#### 3. モデルを作り、評価する

交差検証による学習する処理と、与えられた全データから学習する処理を2パターン書いてます。  
交差検証するときは全体学習をコメントアウト、全体学習するときは交差検証をコメントアウトという使い勝手の悪さが目立ちますね。  

今回は、モデルはLightGBMを用い、交差検証しているもののパラメータチューニング等はしていません。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from models import ModelLGB, Util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb


# LightGBM
def run_lgb(tr_x, tr_y, te_x, te_y, run_fold_name, params, load_path=None):
    lgbm = ModelLGB(run_fold_name, params)
    # 学習
    if load_path is None:
        build_lgb(tr_x, tr_y, lgbm)
        lgbm.save_model()
    else:
        lgbm.load_model()
    
    # 予測
    pred = predict_lgb(te_x, lgbm, params['objective'])

    # 重要度可視化
    plot_lgb_importance(lgbm)

    # 評価
    print(classification_report(te_y, pred, target_names=['0', '1']))
    print(confusion_matrix(te_y, pred))
    acc = accuracy_score(te_y, pred)
    print(acc)

    return acc


def build_lgb(tr_x, tr_y, lgbm, issave=False):
    lgbm.train(tr_x, tr_y)
    if issave:
        lgbm.save_model()
        print('saved model')


def predict_lgb(te_x, lgbm, objective):
    pred = lgbm.predict(te_x)
    if objective == 'multiclass':
        pred = np.argmax(pred, axis=1)
    elif objective == 'binary':
        pred = [1 if p > 0.5 else 0 for p in pred]

    return pred


# 特徴量の重要度を確認
def plot_lgb_importance(lgbm):
    lgb.plot_importance(lgbm.model, height = 0.5, figsize = (4,8))
    plt.show()


def run_cv(train_x, train_y, run_name, params):
    i = 0
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2021)
    for tr_idx, va_idx in skf.split(train_x, train_y):
        run_fold_name = f'{run_name}-{i}'
        tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
        
        score = run_lgb(tr_x, tr_y, va_x, va_y, run_fold_name, params, load_path=None)
        scores.append(score)
        i+=1

    return np.array(scores)


def main():
    features = Util.load('config/features/all.pkl')
    
    # データ取得
    df = pd.read_csv('../data/processed/all.csv')
    train_df = df[df['data']=='train'].reset_index(drop=True)
    test_df = df[df['data']=='test'].reset_index(drop=True)

    train_x = train_df[features]
    train_y = train_df['Survived']
    test_x = test_df[features]

    # LightGBM
    run_name = 'lgb'
    params = {
          'objective' : 'binary', 
          'metric' : 'binary_logloss',
          'verbosity' : -1
    }

    scores = run_cv(train_x, train_y, run_name, params)
    print(scores)
    print(scores.mean())

    # 全体で再度学習
    run_fold_name = f'{run_name}-all'
    lgbm_all = ModelLGB(run_fold_name, params)
    build_lgb(train_x, train_y, lgbm_all)
    pred = predict_lgb(test_x, lgbm_all, params['objective'])
    plot_lgb_importance(lgbm_all)

    left = df.loc[df['data']=='test', 'PassengerId'].reset_index(drop=True)
    right = pd.DataFrame(pred, columns=['Survived'])
    sub_df = pd.concat([left, right], axis=1)
    
    print(sub_df)
    sub_df.to_csv(f'../data/submission/{run_fold_name}.csv', index=False)


if __name__ == "__main__":
    main()

```

<br>
交差検証の結果は以下のようになりました。

- 1回目精度：0.78451
- 2回目精度：0.84511
- 3回目精度：0.78451
- 3回の平均精度：0.80471

Kaggle側の提出結果が以下になります。

[f:id:Noleff:20210531140046p:plain]

<br>
# おわりに

枠組みを作りたかったですが、やる前に作るのはやはり難しいですね。  
こんなのが欲しいなと思った時、随時自作して行こうと思います。

今後作りたいものとしては以下のようなものがあります。

- 学習ログの記録
- 学習ログと特徴量とモデルの関連付け（対応関係がわけわからなくなりそうなため）
- 学習終了後に通知してくれるBotの作成（学習に時間がかかる可能性があるため）
- 交差検証・パラメータチューニング用のクラス

<br>
# Appendix

[こちら](https://github.com/ghmagazine/kagglebook/tree/master/ch04-model-interface/code)を参考にしました。

model.pyとutil.pyは上記リンクと同じです。  
ただし、util.pyに関してはUtilクラスしか、本記事では用いてません（そのため、必要に応じてコメントアウトするところがあるはず）。

#### __init__.py
```python
from .model import Model
from .model_lgb import ModelLGB
from .util import Util
```

#### model_lgb.py
```python
import os
import lightgbm as lgb
from .model import Model
from .util import Util

# LightGBM
class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        isvalid = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y)
        if isvalid:
            lgb_valid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        # num_round = params.pop('num_round')

        # 学習
        if isvalid:
            self.model = lgb.train(params, lgb_train, valid_sets=lgb_valid)
        else:
            self.model = lgb.train(params, lgb_train)


    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


    def save_model(self):
        model_path = os.path.join('../models/lgb', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self):
        model_path = os.path.join('../models/lgb', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)
```

# 参考文献

[https://www.kaggle.com/c/titanic/overview:title]


[https://yolo-kiyoshi.com/2018/12/16/post-951/:title]


[https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science:title]


[https://github.com/ghmagazine/kagglebook:title]