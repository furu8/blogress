ネタ回です。

前回の投稿から少し開きました。
就活と論文のダブルパンチで死んでたわけですが、また引き続き頑張ります。

<br>
# はじめにのはじめに

本記事で出てくる画像は以下から引用してます。

> 株式会社miHoYo
    [http://corp.mihoyo.co.jp/:embed:cite]

> Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc, Food-101 – Mining Discriminative Components with Random Forests, European Conference on Computer Vision, 2014.
    [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/:embed:cite]


<br>
# はじめに


今回の内容は、前回の投稿内容の続き的な立ち位置になります。

#### 趣旨

原神というゲームがあります。  
最近、週一でやっているかどうかくらいのペースでしかやっていなかったります。    
ぶっちゃけ、もう飽きてる節がありますが、たまに惰性でやってます。  
本記事の内容は、タイトルで察しってください、と言いたいところですが趣旨を説明します。  
<br>
ゲームに限りませんが、序盤から主人公と一緒にいて、何かと手助けしてくれるマスコット的なキャラクターがいるかと思います。  
原神では、それがパイモンです。  
ただ、主人公との出会いが釣りをしてたら釣れたという経緯があり、**公式公認で非常食扱い**されてます。  
<br>

[f:id:Noleff:20210520204741j:plain:w400:h250]

<br>
かわいそうなので、**愛（AI）の力でパイモンを救ってみよう**、というわけです。

<br>
####  データ

画像は二種類用意します。  
一種類目は、もちろんパイモンの画像です。
Twitterから主に集めた画像を顔画像だけトリミングしたものを用意してます。  

[https://noleff.hatenablog.com/entry/2021/02/22/004156:embed:cite]

<br>
二種類目は、食べ物の画像です。
これは[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)という101種類の食べ物の画像が1000枚ずつ、合計101000枚あるデータセットになります。

これらの画像を学習し、分類するモデルを作ることが本記事のゴールです。

<br>
# 環境

|  言語・ライブラリ  |  バージョン  | 
| ---- | ---- | 
|  python  |  3.7.9  | 
|  pandas  |  1.2.0  | 
|  numpy |  1.19.2  | 
|  scikit-learn  |  0.23.2  |   
|  tensorflow  |  2.0.0  |  

なお、学習にはGPU（GTX1060 6GB）使っています。  

<br>
# いざ、救う

#### 二クラス分類プログラム

pathは任意

```python

import glob as gb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib

# 画像の読み込み
def load_image_npy(load_path, isdir=False):
    if isdir:
        return np.array([np.load(path, allow_pickle=True) for path in gb.glob(load_path)])
    else:
        return np.load(load_path, allow_pickle=True)


# 学習データとテストデータにわける
def make_train_test_data(image1, image2):
    X = np.concatenate([image1, image2])
    y = np.array([0] * len(image1) + [1] * len(image2)) # face:0, food:1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=2021)

    return X_train, X_test, y_train, y_test


# モデル構築
def build_cnn_model():
    model = Sequential()

    # 入力画像 64x64x3 (縦の画素数)x(横の画素数)x(チャンネル数)
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def learn_model(model, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        test_size=0.25, 
                                                        shuffle=True, 
                                                        random_state=2021)

    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=1.0e-3, 
                                patience=20, 
                                verbose=1)
    hist = model.fit(X_train, y_train, 
                batch_size=1000, 
                verbose=2, 
                epochs=100, 
                validation_data=(X_val, y_val), 
                callbacks=[early_stopping])
    
    return hist


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score


def predict_model(model, X_test):
    pred_y = model.predict(X_test, batch_size=128)

    # pred_y = np.argmax(pred_y, axis=1)

    return pred_y


def save_model(path, model):
    model.save(path)
    print('saved model: ', path)


# 評価系のグラフをプロット
def plot_evaluation(eval_dict, key1, key2, ylabel, save_path=None):
    plt.figure(figsize=(10,7))
    plt.plot(eval_dict[key1], label=key1)
    plt.plot(eval_dict[key2], label=key2)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    
def plot_cmx_heatmap(cmx, labels, save_path=None):
    df_cmx = pd.DataFrame(cmx, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cmx, annot=True, fmt='d')
    plt.ylim(0, len(labels)+1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def main():
    # GPUの動作確認
    # print(device_lib.list_local_devices())
    labels = ['food', 'face']
    save_file_name = 'bin.png'

    face_image = load_image_npy('D:/Illust/Paimon/interim/npy_face_only/paimon_face_augmentation.npy')
    food_image = load_image_npy('D:/OpenData/food-101/interim/npy_food-101.npy')
    food_image = food_image[np.random.choice(food_image.shape[0], 10000, replace=False), :] # food_image101000枚の画像からランダムに10000枚抽出
    print(face_image.shape)
    print(food_image.shape)

    X_train, X_test, y_train, y_test = make_train_test_data(face_image, food_image)

    score_list = []
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2021)
    for train_idx, val_idx in kf.split(X_train, y_train):
        train_x, val_x = X_train[train_idx], X_train[val_idx]
        train_y, val_y = y_train[train_idx], y_train[val_idx]

        model = build_cnn_model()
        hist = learn_model(model, train_x, train_y)

        score = evaluate_model(model, val_x, val_y)
        pred_y = predict_model(model, val_x)

        pred_y = [1 if y > 0.9 else 0 for y in pred_y.flatten()]

        score_list.append(score[1])
        print(classification_report(val_y, pred_y, target_names=labels))
        cmx = confusion_matrix(val_y, pred_y)
        print(cmx)

        # plot_cmx_heatmap(cmx, labels)
    
    print(score_list)
    print(np.array(score_list).mean())

    # 再度学習
    model = build_cnn_model()
    hist = learn_model(model, X_train, y_train)

    plot_evaluation(hist.history, 'loss', 'val_loss', 'loss', 'figures/loss_'+save_file_name)
    plot_evaluation(hist.history, 'accuracy', 'val_accuracy', 'accuracy', 'figures/acc_'+save_file_name)

    y_pred = predict_model(model, X_test)
    y_pred = [1 if y > 0.9 else 0 for y in y_pred.flatten()]

    print(classification_report(y_test, y_pred, target_names=labels))
    cmx = confusion_matrix(y_test, y_pred)
    print(cmx)

    plot_cmx_heatmap(cmx, labels, save_path='figures/cmx_'+save_file_name)


if __name__ == "__main__":
    main()
```

<br>

#### 全体の流れ

<br>
1. 画像読み込み

  - パイモン画像：8338枚（オーグメンテーション済み）
  - 食べ物の画像：10000枚

<br>
2. 学習データとテストデータに分割

  学習するための学習データと最終的に評価するテストデータにわけます（全体データ数の25%をテストデータに）  
  このとき、正解ラベルも加えます。

<br>
3. 交差検証

  学習データの中から3分割して交差検証します。  
  予測値は0~1の範囲で出力されます。0に近いほどパイモン、1に近いほど食べ物の画像となります。予測値から0.9より大きな値は1、0.9以下の値は0となるようにしました。

  精度と損失の学習曲線を可視化し、適合率、再現率、F値、精度、混同行列により定量的にモデルを評価します。  
  このとき、3分割した平均精度も算出します

<br>
4. 学習データ全体で学習

  本来なら交差検証でチューニング等済ませてから全体で学習しますが、今回は交差検証後にそのまま全体で学習してます。  
  学習が高速に行えたので、どのくらい精度にブレがあるか調べたかったため交差検証しているだけです。

<br>
5. 結果を出力・保存

  こちらも交差検証同様、学習データ全体の精度と損失の学習曲線を可視化し、
  学習データ全体の適合率、再現率、F値、精度、混同行列により定量的にモデルを評価します。

<br>
#### パイモン画像のオーグメンテーション

4000枚ほどの画像を約2倍の8000枚ほどまで水増ししました。
詳しいオーグメンテーションのコードは後述します。

以下、行ったオーグメンテーション

- ランダムに±指定した角度の範囲で回転 
- ランダムに±指定した横幅に対する割合の範囲で左右方向移動
- ランダムに±指定した縦幅に対する割合の範囲で左右方向移動
- ランダムにズームする
- 水平方向に入力をランダムに反転
- 垂直方向に入力をランダムに反転

これらの処理をあらかじめ行っておき、保存したものを最初に読み込んでます。

<br>
#### 二クラス分類結果

かなり、見にくいですが……。  
学習曲線は学習中の検証データに対する精度と損失です。  
最終的なテストデータに対する評価が混同行列になります。

|  学習曲線（精度）  |  学習曲線（損失）  |   混同行列  | 
| ---- | ---- | ---- |
| [f:id:Noleff:20210520191746p:plain:w300:h150] | [f:id:Noleff:20210520191759p:plain:w300:h150] | [f:id:Noleff:20210520192002p:plain:w300:h150] |

<br>
**パイモンは食べ物（非常食）でないことを100%分類できました！**  

これは完全に救ってしまったのではなかろうか……  
と、言いたいところですが交差検証評価するデータを変えると100%でないときがあります（Appendix参照）。

加えてCNNも対して工夫していないにも関わらず、この高い精度です。  
つまり、ぶっちゃけ機械学習のタスクとして二クラス（パイモンと食べ物）を分類することは、そう難しくありません。

実はここまでコードを書くのは割と一瞬だったので、もう少し難しいタスクに挑戦します。


<br>
#### 多クラス分類プログラム

ここからが本番です。  
二クラス分類プログラムと同じ関数等ありますが、すべて掲載します。

```python
import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import re

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


def load_image_npy(load_path, isdir=False):
    if isdir:
        return np.array([np.load(path, allow_pickle=True) for path in gb.glob(load_path)])
    else:
        return np.load(load_path, allow_pickle=True)


def make_train_test_data(image1, image2, labels):
    X = np.concatenate([image1, image2], axis=0)

    label_list = [0] * len(image1)
    for i in range(len(labels)):
        label_list += [i+1] * 1000 
    y = np.array(label_list)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=2021)

    return X_train, X_test, y_train, y_test


# モデル構築
def build_cnn_model():
    model = Sequential()

    # 入力画像 64x64x3 (縦の画素数)x(横の画素数)x(チャンネル数)
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # model.add(Dense(3200, activation='relu',kernel_initializer='he_normal'))  
    # model.add(Dense(800, activation='relu', kernel_initializer='he_normal'))  
    # model.add(Dense(120, activation='relu', kernel_initializer='he_normal')) 
    # model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def build_imagenet():
    base_model = InceptionV3(weights='imagenet', 
                            include_top=False, 
                            input_tensor=Input(shape=(128, 128, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(15, activation='softmax')(x)

    model = Model(base_model.input, predictions)

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def learn_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=1.0e-3, 
                                patience=20, 
                                verbose=1)
    hist = model.fit(X_train, y_train, 
                    batch_size=32, 
                    verbose=2, 
                    steps_per_epoch=X_train.shape[0] // 32,
                    epochs=100, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping]
                    )
                    
    return hist

def learn_model_generator(model, X_train, y_train, X_val, y_val, tr_datagen, va_datagen):
    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=1.0e-3, 
                                patience=20, 
                                verbose=1)

    hist = model.fit_generator(tr_datagen.flow(X_train, y_train, batch_size=32), 
                                verbose=2, 
                                steps_per_epoch=X_train.shape[0] // 32,
                                epochs=100, 
                                validation_data=(X_val, y_val), 
                                callbacks=[early_stopping]
                                )
    return hist


# 画像の水増し
def make_datagen(rr=30, wsr=0.1, hsr=0.1, zr=0.2, val_spilit=0.2, hf=True, vf=True):
    datagen = ImageDataGenerator(
            featurewise_center = False,            # データセット全体で，入力の平均を0にするかどうか
            samplewise_center = False,             # 各サンプルの平均を0にするかどうか
            featurewise_std_normalization = False, # 入力をデータセットの標準偏差で正規化するかどうか
            samplewise_std_normalization = False,  # 各入力をその標準偏差で正規化するかどうか
            zca_whitening = False,                 # ZCA白色化を適用するかどうか
            rotation_range = rr,                   # ランダムに±指定した角度の範囲で回転 
            width_shift_range = wsr,               # ランダムに±指定した横幅に対する割合の範囲で左右方向移動
            height_shift_range = hsr,              # ランダムに±指定した縦幅に対する割合の範囲で左右方向移動
            zoom_range = zr,                       # 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
            horizontal_flip = hf,                  # 水平方向に入力をランダムに反転するかどうか
            vertical_flip = vf,                    # 垂直方向に入力をランダムに反転するかどうか
            validation_split = val_spilit          # 検証のために予約しておく画像の割合（厳密には0から1の間）
        )
    
    return datagen


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score


def predict_model(model, X_test):
    y_pred = model.predict(X_test, batch_size=128)

    # y_pred = np.argmax(y_pred, axis=1)

    return y_pred


def save_model(path, model):
    model.save(path)
    print('saved model: ', path)


# 評価系のグラフをプロット
def plot_evaluation(eval_dict, key1, key2, ylabel, save_path=None):
    plt.figure(figsize=(10,7))
    plt.plot(eval_dict[key1], label=key1)
    plt.plot(eval_dict[key2], label=key2)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def main():
    # GPUの動作確認
    # print(device_lib.list_local_devices())

    face_images = load_image_npy('D:/Illust/Paimon/interim/npy_face_only/paimon_face.npy')
    food_images = load_image_npy('D:/OpenData/food-101/interim/npy_food-101_64/npy_food-101_64.npy')
   
    # print(face_images)
    # print(food_images)
    print(face_images.shape)
    print(food_images.shape)
   
    # Food-101のデータがディレクトリでわけられているのでディレクトリ名=ラベルとしている
    labels = [re.split('[\\\.]',path)[-2] for path in gb.glob('D:/OpenData/food-101/interim/npy_food-101_64/food/*')]

    X_train, X_test, y_train, y_test = make_train_test_data(face_images, food_images, labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=2021)

    labels = ['paimon'] + labels

    # ラベルをOne-Hotに変換
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    # 学習
    model = build_cnn_model()
    # model = build_imagenet()

    hist = learn_model(model, X_train, y_train, X_val, y_val)
    # train_datagen = make_datagen()
    # valid_datagen = ImageDataGenerator()
    # hist = learn_model_generator(model, X_train, y_train, X_val, y_val, train_datagen, valid_datagen)
    
    save_file_name = 'same_bin.png' #  実験ごとに適宜変える
    plot_evaluation(hist.history, 'loss', 'val_loss', 'loss', 'figures/loss_'+save_file_name)
    plot_evaluation(hist.history, 'accuracy', 'val_accuracy', 'accuracy', 'figures/acc_'+save_file_name)

    y_pred = predict_model(model, X_test)
    y_pred = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred, target_names=labels))

    cmx = confusion_matrix(y_test, y_pred)
    print(cmx)

    df_cmx = pd.DataFrame(cmx, index=labels, columns=labels)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cmx, annot=True, fmt='d')
    plt.ylim(0, len(labels)+1)
    # plt.show()
    plt.savefig('figures/cmx_'+save_file_name)

if __name__ == "__main__":
    main()
```

<br>
## 補足説明

### 二クラス分類プログラムとの変更点

多クラス分類プログラムでは、オーグメンテーションをこのプログラム内でやってます。  
交差検証は一回の学習の処理が重い（長い）のでやっていないです。  
多クラス分類 = 15クラス分類のモデルを作ります。

- パイモン1クラス
- 食べ物14クラス

参考までに、下表にパイモンと食べ物14種類を並べました（ツナタルタルだけよくわかりませんが）。

|  food(en)  |  food(ja)  |   image  | 
| ---- | ---- | ---- | 
| paimon | パイモン | [f:id:Noleff:20210520172533p:plain:w100:h100] |
|  baby_back_ribs | スペアリブ |  [f:id:Noleff:20210516185144j:plain:w100:h100]  | 
|  cup_cakes | カップケーキ |  [f:id:Noleff:20210519091217j:plain:w100:h100]  | 
|  dumplings  | 小籠包 |  [f:id:Noleff:20210519091351j:plain:w100:h100]  | 
|  edamame | 枝豆 |  [f:id:Noleff:20210519091449j:plain:w100:h100]  | 
|  guacamole  | ワカモレ | [f:id:Noleff:20210520195217j:plain:w100:h100] |
|  miso_soup  | 味噌汁 |  [f:id:Noleff:20210519091721j:plain:w100:h100]  | 
|  mussels  | ムール貝 |  [f:id:Noleff:20210519091816j:plain:w100:h100]  | 
|  nachos  | ナチョス |  [f:id:Noleff:20210519091935j:plain:w100:h100]  | 
|  oysters | 牡蠣 |  [f:id:Noleff:20210519092020j:plain:w100:h100]  | 
|  pancakes  | パンケーキ |  [f:id:Noleff:20210519092135j:plain:w100:h100]  | 
|  ramen  | ラーメン |  [f:id:Noleff:20210519092423j:plain:w100:h100]  | 
|  sushi  | 寿司 |  [f:id:Noleff:20210519092532j:plain:w100:h100]  | 
|  tuna_tartare  | ツナタルタル？ |  [f:id:Noleff:20210519092742j:plain:w100:h100]  | 
|  waffles | ワッフル |  [f:id:Noleff:20210519092819j:plain:w100:h100]  | 

<br>
### 実験概要

全部で6種類実験したので、その結果を載せます。

1. 二クラス分類と同じネットワークアーキテクチャ
2. 二クラス分類と同じネットワークアーキテクチャ（オーグメンテーション有り）
3. CNNの層を増加
4. CNNの層を増加（オーグメンテーション有り）
5. Inception-v3
6. Inception-v3（オーグメンテーション有り）

<br>
### 1. 二クラス分類と同じネットワークアーキテクチャ

上記プログラムがこれになります。  
ただし、出力15クラスなので出力層のノード数を15にし、活性化関数をシグモイド関数からソフトマックス関数にしてます。  
同様に多クラス分類なので、損失関数をbinary_crossentropyからcategorical_crossentropyにしてます。  
それ以外は同じです。

```python
# モデル構築
def build_cnn_model():
    model = Sequential()

    # 入力画像 64x64x3 (縦の画素数)x(横の画素数)x(チャンネル数)
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(15, activation='softmax')) # 15クラスで出力してほしいので15に、活性化関数をソフトマックス関数に

    model.compile(
        loss='categorical_crossentropy', # 多クラス分類なので損失関数はcategorical_crossentropyに
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model

```

### 2. 二クラス分類と同じネットワークアーキテクチャ（オーグメンテーション有り）

新しく、以下の関数を定義します。  
これらのコードで画像の水増し及び、学習を行います。

```python
def learn_model_generator(model, X_train, y_train, X_val, y_val, tr_datagen, va_datagen):
    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=1.0e-3, 
                                patience=20, 
                                verbose=1)

    hist = model.fit_generator(tr_datagen.flow(X_train, y_train, batch_size=32), 
                                verbose=2, 
                                steps_per_epoch=X_train.shape[0] // 32,
                                epochs=100, 
                                validation_data=(X_val, y_val), 
                                callbacks=[early_stopping]
                                )
    return hist


# 画像の水増し
def make_datagen(rr=30, wsr=0.1, hsr=0.1, zr=0.2, val_spilit=0.2, hf=True, vf=True):
    datagen = ImageDataGenerator(
            featurewise_center = False,            # データセット全体で，入力の平均を0にするかどうか
            samplewise_center = False,             # 各サンプルの平均を0にするかどうか
            featurewise_std_normalization = False, # 入力をデータセットの標準偏差で正規化するかどうか
            samplewise_std_normalization = False,  # 各入力をその標準偏差で正規化するかどうか
            zca_whitening = False,                 # ZCA白色化を適用するかどうか
            rotation_range = rr,                   # ランダムに±指定した角度の範囲で回転 
            width_shift_range = wsr,               # ランダムに±指定した横幅に対する割合の範囲で左右方向移動
            height_shift_range = hsr,              # ランダムに±指定した縦幅に対する割合の範囲で左右方向移動
            zoom_range = zr,                       # 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
            horizontal_flip = hf,                  # 水平方向に入力をランダムに反転するかどうか
            vertical_flip = vf,                    # 垂直方向に入力をランダムに反転するかどうか
            validation_split = val_spilit          # 検証のために予約しておく画像の割合（厳密には0から1の間）
        )
    
    return datagen
```

上記のコードで学習するために、main関数を書き換えます。

```python
# before
# hist = learn_model(model, X_train, y_train, X_val, y_val) 

# after
train_datagen = make_datagen()
valid_datagen = ImageDataGenerator()
hist = learn_model_generator(model, X_train, y_train, X_val, y_val, train_datagen, valid_datagen)
```

以下、オーグメンテーション無しのときはbeforeで学習し、有りのときはafterで学習します。

<br>
### 3. CNNの層を増加

以下のようにbuild_cnn_model関数を書き換えます。

- もう一層分、隠れ層を追加
- 全結合層のあと、徐々に15クラスまで次元を落としていく
- 過学習を防ぐDropout層を追加


```python
def build_cnn_model():
    model = Sequential()

    # 入力画像 64x64x3 (縦の画素数)x(横の画素数)x(チャンネル数)
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(3200, activation='relu',kernel_initializer='he_normal'))  
    model.add(Dense(800, activation='relu', kernel_initializer='he_normal'))  
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal')) 
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model
```

<br>
### 4. CNNの層を増加（オーグメンテーション有り）

3\. CNNの層を増加のbuild_cnn_model関数でモデルを構築し、 2\. 二クラス分類と同じネットワークアーキテクチャ（オーグメンテーション有り）のコードで学習します。

<br>
### 5. Inception-v3

Inception-v3はGoogleによって開発された、画像の1000クラス分類を行うよう学習された深層学習モデルです。  
この学習済みモデルを転移学習して予測します。

↓ Inception-v3に関する論文
> C\. Szegedy, V. Vanhoucke, S. Ioffe, and J. Shlens. Rethinking the inception architecture for computer vision. In Proc. of CVPR, 2016.

inception-v3を用いて、food-101のデータセット101クラス分類タスクとして、精度約82%出したリポジトリが以下になります。  
コード書く上で参考にさせていただきました。

[https://github.com/stratospark/food-101-keras:embed:cite]

<br>
build_cnn_model関数の代わりに以下の関数を定義します。

```python
def build_imagenet():
    base_model = InceptionV3(weights='imagenet', 
                            include_top=False, 
                            input_tensor=Input(shape=(128, 128, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(15, activation='softmax')(x)

    model = Model(base_model.input, predictions)

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
```

<br>
今まで画像のピクセルを64x64で学習させていましたが、inception-v3では128x128で学習させます。  
これは、incepiton-v3がピクセルの幅と高さを75以上にしないといけないためです。

[https://keras.io/api/applications/inceptionv3/:embed:cite]

<br>
main関数で画像を読み込んだ直後に以下のコードを入れれば、画像を128x128にリサイズできます。  
できますが、かなり強引な手法なのでメモリは食いまくります。当方、16GBでギリといった感じです。

```python
resize_num = 128
face_images = np.array([cv2.resize(face_image, (resize_num,resize_num)) for face_image in face_images])
food_images = np.array([cv2.resize(food_image, (resize_num,resize_num)) for food_image in food_images])
```

<br>
 build_imagenet関数呼び出し時を書き換えます。

```python
# model = build_cnn_model()
model = build_imagenet()
```

<br>
### 6.  Inception-v3（オーグメンテーション有り）

5\.  Inception-v3のbuild_cnn_model関数でモデルを構築し、 2\. 二クラス分類と同じネットワークアーキテクチャ（オーグメンテーション有り）のコードで学習します。

<br>
## 多クラス分類結果

1. 二クラス分類と同じネットワークアーキテクチャ
2. 二クラス分類と同じネットワークアーキテクチャ（オーグメンテーション有り）
3. CNNの層を増加
4. CNNの層を増加（オーグメンテーション有り）
5. Inception-v3
6. Inception-v3（オーグメンテーション有り）

|   |  学習曲線（精度）  |  学習曲線（損失）  |   混同行列  | 
| ---- | ---- | ---- |  ---- | 
| 1 | [f:id:Noleff:20210520190438p:plain:w300:h150] | [f:id:Noleff:20210520190526p:plain:w300:h150] | [f:id:Noleff:20210520190346p:plain:w300:h150] |
| 2 | [f:id:Noleff:20210520190718p:plain:w300:h150] | [f:id:Noleff:20210520190852p:plain:w300:h150] | [f:id:Noleff:20210520190834p:plain:w300:h150] |
| 3 | [f:id:Noleff:20210520190921p:plain:w300:h150] | [f:id:Noleff:20210520190950p:plain:w300:h150] | [f:id:Noleff:20210520191013p:plain:w300:h150] |
| 4 | [f:id:Noleff:20210520191028p:plain:w300:h150] | [f:id:Noleff:20210520191103p:plain:w300:h150] | [f:id:Noleff:20210520191119p:plain:w300:h150] |
| 5 | [f:id:Noleff:20210520191212p:plain:w300:h150] | [f:id:Noleff:20210520191231p:plain:w300:h150] | [f:id:Noleff:20210520191246p:plain:w300:h150] |
| 6 | [f:id:Noleff:20210520191259p:plain:w300:h150] | [f:id:Noleff:20210520191321p:plain:w300:h150] | [f:id:Noleff:20210520191334p:plain:w300:h150] |


<br>
### 1と2

1は精度47%でした。全体の半分以上間違ってるということになります。    
2のオーグメンテーション有りにすることでなんとか、精度57%になりました。

<br>
### 3と4

3は1と同じ精度47%でした。  
4の方は62%でした。2より少しだけ精度が上がった程度です。    
今回の工夫はあまり意味なったかもしれません。

<br>
### 5と6

5は劇的に伸びて、精度82%まで行きました。  
6はさらに伸びる！　かと思いきや同じ82%でした。
しかし、パイモンに関しては適合率、再現率ともに1.0なので、誤判定も見逃しもせず判定できたことになります。1~5ではこうはなりませんでした。  
つまり、**パイモンだけは完全に分類しきったので、救ったと言っても過言ではないでしょう！**   
<center>(o゜ー゜o)??</center>

<br>
なお、食べ物はムール貝とスペアリブを間違えたり、パンケーキとワッフルを間違えたりしていることが多かったです。  

<br>
## ぷち考察

6でパイモンは完全に分類しきりましたが、食べ物同士は結構間違ってます。  
Food-101のデータセットは食べ物が複数写っている画像もあります。その当たりを考慮してモデルを作成していないのが原因の一つかと思われます。  
極端に言えば、人が写ってる画像もありました。

また、Food-101のデータセットを使って、ラベル付けの間違いを見つけるブログもあるので、Food-101内のラベリングミスがあるのかもしれません（怪しいなと思う画像はいくつかありました）。

[https://www.kccs.co.jp/labellio/blog/2019/07/rps.html:embed:cite]

<br>
# おわりに

甘雨が好きです。  
パイモンはそこまで好きじゃないです。

<br>
# Appendix

## 二クラス分類交差検証結果

3回の平均精度：0.99978246

- 1回目
  - 精度：0.99978244
  - 混同行列
```
[[2099    0]
 [   1 2497]]
```

- 2回目
  - 精度：0.99956495
  - 混同行列
```
[[2099    0]
 [   3 2495]]
```

- 3回目
  - 精度：1.0
  - 混同行列
```
[[2099    0]
 [   0 2498]]
```

# 参考文献

[http://corp.mihoyo.co.jp/:title]

[http://corp.mihoyo.co.jp/policy/guideline1.html:title]

[https://genshin.mihoyo.com/ja/game:title]

[https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/:title]

[https://github.com/stratospark/food-101-keras:title]

[https://keras.io/api/applications/inceptionv3/:title]

[https://www.kccs.co.jp/labellio/blog/2019/07/rps.html:title]

Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc, Food-101 – Mining Discriminative Components with Random Forests, European Conference on Computer Vision, 2014.

C\. Szegedy, V. Vanhoucke, S. Ioffe, and J. Shlens. Rethinking the inception architecture for computer vision. In Proc. of CVPR, 2016.


