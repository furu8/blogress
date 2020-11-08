# はじめに

今回の記事では、Kagglerランカー達がこぞって使ってるという「LightGBM」なるものを使ってプログラミングしたことをまとめていきます。  

# LightGBMとは

LightGBMとは決定木アルゴリズムを応用したアルゴリズムです。よくXGBoostと比較されるのを目にします。

詳しくは以下のサイトをご参考ください。決定木、アンサンブル学習、勾配ブースティング（LightGBMやXGBoost）の順に、非常にわかりやすくまとめられています。

[https://www.codexa.net/lightgbm-beginner/:embed:cite]

# プログラミング

## データ

実際に手を動かしていきます。まず、肝心のデータですが、今回は**高松市**の気象データを[気象庁](https://www.data.jma.go.jp/obd/stats/etrn/index.php)の方からダウンロードさせていただきました。    
気象データの期間は2019-01-01から2020-08-29までです。

## 目標

今回の目標は「一時間単位に高松市の気温を予測する」こととします。そのためダウンロードした気象データは一日単位ではなく、一時間単位の気象データを収集しました。

## プログラム

先に関数を定義しておきます。  

```python
# データ読み込み
def read_df(path):
    df = pd.read_csv(path, sep=',')
    return df

# データ抽出
def extract_train_test_data(df):
    train_df = df[df['year'] == 2019]
    test_df = df[(df['year'] == 2020)]

    return train_df, test_df

# 学習、テストデータ作成
def make_train_test_data(df, col_list):
    train_df, test_df = extract_train_test_data(df)
    
    # 学習用、テスト用にデータを分割
    # X_train = train_df[col_list]
    X_train = train_df.drop(['datetime', 'temperature'], axis=1)
    y_train = train_df['temperature']
    
    # X_test = test_df[col_list]
    X_test = test_df.drop(['datetime', 'temperature'], axis=1)
    y_test = test_df['temperature']

    return X_train, y_train, X_test, y_test
```

まず、データを読み込んで表示します。read_df関数の引数のファイル名は任意です。

```python
# データ読み込み
df = read_df('20190101-20200829-takamatsu.csv')
df
```

気象庁のデータをそのまま使うと、pandasを使う上で不都合が多いので、以下のように整形しています。約14500レコードになりました。

[f:id:Noleff:20200901185420p:plain]

カラムそれぞれの意味は以下の表にまとめます。

|  カラム名 |  意味  |
| ---- | ---- |
|  datetime  |  日付  |
|  temperature  |  気温  |
|  pressure |  気圧  |
|  relative_humidity  |  相対湿度  |
|  wind_speed  |  風速  |
|  rainfall  |  降水量  |
|  sea_lavel_pressure |  海面気圧  |
|  day_length  |  日照時間  |
|  year  |  年  |
|  month |  月  |
|  day  |  日  |

次に、気象データを学習用とテスト用に分割します。今回は2019の気象データを学習データ、2020の気象データをテストデータとしました。  
col_listは目的変数である気温を予測するために使用する説明変数のカラムです。**今回は使用していません。**  特徴量選択や特徴量エンジニアリングについて触れず、気温と日付以外のすべてのカラムを説明変数（特徴量）とします。

```python
# 学習、検証データ作成
col_list = ['relative_humidity', 'pressure', 'day_length', 'month', 'day', 'hour']
X_train, y_train, X_test, y_test = make_train_test_data(df, col_list)
```

LightGBM用にデータをセットし、学習データで回帰モデルを作成、テストデータで予測をします。  
実際の2020-01-01から2020-08-29の気温データと、予測した結果を連結し、predicted_dfにまとめます。  
評価基準はrmse（Root Mean Square Error）としています。

```python
# LightGBM用のデータセット
lgb_train = lgb.Dataset(X_train, y_train)
lgb.test = lgb.Dataset(X_test, y_test)

# 評価基準 
params = {'metric' : 'rmse'}

# 回帰モデル作成
gbm = lgb.train(params, lgb_train)

# 予測
test_predicted = gbm.predict(X_test)
predicted_df = pd.concat([y_test.reset_index(drop=True), pd.Series(test_predicted)], axis = 1)
predicted_df.columns = ['true', 'pred']
predicted_df
```

[f:id:Noleff:20200901185451p:plain]


予測値と実測値の散布図を作成してみます。散布図と一緒に赤色で y = x の一次関数も描画しました。赤色にデータが近いほど良い結果が出ていると言えます。  

```python
# 予測値と実測値の散布図
RMSE = np.sqrt(mean_squared_error(predicted_df['true'], predicted_df['pred']))

plt.figure(figsize = (4,3))
plt.scatter(predicted_df['true'], predicted_df['pred'])
plt.plot(np.linspace(0, 40), np.linspace(0, 40), 'r-')

plt.xlabel('True Temperature')
plt.ylabel('Predicted Temperature')
plt.text(0.1, 30, 'RMSE = {}'.format(str(round(RMSE,3))), fontsize=10)

plt.show()
```

[f:id:Noleff:20200901190031p:plain]

今度は予測値と実測値をそれぞれ折れ線図として作成してみます。横軸は2020-01-01 01:00:00 からの経過時間（一時間単位）です。

```python
# 予測値と実測値の折れ線図
plt.figure(figsize=(16,4))
plt.plot(predicted_df['true'], label='true')
plt.plot(predicted_df['pred'], label='pred')

plt.xlabel('time')
plt.ylabel('temperature')
# plt.xlim(0,1000)
plt.legend()

plt.show()
```

[f:id:Noleff:20200901190341p:plain]

大体うまくいけているのではないでしょうか。  
もう少しクローズアップしてデータを見てみます。上記のプログラムにplt.xlimメソッドを使って横軸に制限をかけたときの折れ線図を示します。  
0-1000の図で、横軸が500から600にかけてのところで予測値と実測値がずれていることがわかります。

- 0-100

[f:id:Noleff:20200901190643p:plain]

- 0-1000

[f:id:Noleff:20200901190715p:plain]

最後に有効な特徴量が何だったのか見てみます。  
dayやmonthなどが上位に来ています。

```python
# 特徴量の重要度
lgb.plot_importance(gbm, height = 0.5, figsize = (4,8))
plt.show()
```

[f:id:Noleff:20200901192312p:plain]

# 所感

思ったよりもうまくできたので正直驚きました。Lightと名前につくだけあってとても高速です。  
ハイパーパラメータをチューニングしたり、特徴量を選択してないため実装も簡単です。  
検討すれば精度はさらに向上すると思います。

# 参考文献

[https://www.data.jma.go.jp/obd/stats/etrn/index.php:title]

[https://rightcode.co.jp/blog/information-technology/lightgbm-useful-for-kaggler:title]

[https://www.codexa.net/lightgbm-beginner/:title]

[https://rin-effort.com/2019/12/29/machine-learning-6/:title]

[https://note.com/___n_h__n__/n/na69a79719d56:title]

# Appendix

今回のデータのペアプロットと相関係数を以下に添付します。  
このように可視化することで、目的変数との相関を見比べれば、特徴量選択を決める一つの指標になると思います。

- ペアプロット

[f:id:Noleff:20200907155908p:plain]

- 相関

[f:id:Noleff:20200907155945p:plain]














