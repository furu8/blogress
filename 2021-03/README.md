
# 概要
## フォルダ構成

```
├── data                      <- データ関連
│   ├── interim               <- 作成途中のデータ
│   ├── processed             <- 学習に使うデータ
│   ├── raw                   <- 生データ
│   │   ├── test.csv
│   │   ├── train.csv
│   ├── submittion            <- 提出用データ
│   │   ├── sample_submission.csv
├── scripts                   <- プログラム類
│   ├── generate.py           <- データ作成（生データを分析しやすい整然データに→interimに保存）
│   ├── analyze.py            <- 分析用スクリプト
│   ├── run.py                <- 学習用スクリプト
│   ├── models                <- モデル関連クラス
│   │   ├── model.py          <- モデル基底クラス
│   │   ├── model_lgb.py      <- LightGBMクラス
│   │   ├── util.py           <- 汎用的処理クラス
│   ├── config                <- 汎用的処理クラス
│   │   ├── features          <- 特徴量のカラム群
│   │   ├── params            <- ハイパーパラメータ（コードに直接書いた方が良いかも）
├── models                    <- 作成したモデル保存（モデル名_日付_精度）
├── logs                      <- 作成したモデル保存（日付_特徴量）
```