# %%
import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import re
import joblib

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import AveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img 

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib

# %%
def load_npy_image(load_path):
    return np.load(load_path, allow_pickle=True)

def make_dataset(daisy, dandelion, rose, sunflower, tulip, fruit_vegetable, fruit_vegetable_labels):
    # 画像
    flowers = np.concatenate([daisy, dandelion], axis=0)
    flowers = np.concatenate([flowers, rose], axis=0)
    flowers = np.concatenate([flowers, sunflower], axis=0)
    flowers = np.concatenate([flowers, tulip], axis=0)
    X = np.concatenate([flowers, fruit_vegetable], axis=0)

    # ラベル
    labels = [0] * len(daisy) + [1] * len(dandelion) + [2] * len(rose) + [3] * len(sunflower) + [4] * len(tulip) # flowers label
    for i in range(len(fruit_vegetable_labels)):
        labels += [i+5] * 100  # 各画像ごとに100枚ずつ fruit_vegetable label
    y = np.array(labels)

    return X, y

def build_model(out_shape):
    # https://github.com/stratospark/food-101-keras
    # https://note.com/matsukoutennis/n/nfaa6b86ddf15
    # https://qiita.com/koshian2/items/b2d9c03ece95cf5f280a
    base_model = InceptionV3(weights='imagenet', 
                            include_top=False, 
                            input_tensor=Input(shape=(128, 128, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(out_shape, activation='softmax')(x)

    model = Model(base_model.input, predictions)

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# モデル構築
# def build_model(out_shape):
#     '''
#     初期化 (initializer)
#     Glorotの初期化法:sigmoid関数やtanh関数
#     Heの初期化法:ReLU関数
#     https://ichi.pro/zukai-10-ko-no-cnn-a-kitekucha-164752979288397
#     '''
#     model = Sequential()

#     # 入力画像 128x128x3 (縦の画素数)x(横の画素数)x(チャンネル数)
#     model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal', input_shape=(128, 128, 3)))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#     model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#     model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.4))

#     model.add(Flatten())
#     model.add(Dense(2560, activation='relu',kernel_initializer='he_normal'))
#     model.add(Dropout(0.2))  
#     model.add(Dense(640, activation='relu', kernel_initializer='he_normal'))  
#     model.add(Dense(128, activation='relu', kernel_initializer='he_normal')) 
#     model.add(Dropout(0.2))
#     model.add(Dense(out_shape, activation='softmax'))

#     model.compile(
#         loss='categorical_crossentropy',
#         optimizer='adam',
#         metrics=['accuracy']
#     )

#     model.summary()

#     return model

# 画像の水増し
def make_datagen(rr=30, wsr=0.1, hsr=0.1, zr=0.2, val_spilit=0.2, hf=True, vf=True):
    datagen = ImageDataGenerator(
            # https://keras.io/ja/preprocessing/image/
            # rescale = 1./255,                      # スケーリング　
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

def learn_model(model, X_train, y_train, X_val, y_val):
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    tr_datagen = make_datagen()       # 学習データだけ水増し
    va_datagen = ImageDataGenerator() # 検証データは水増ししない

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

def predict_model(model, X_test):
    y_pred = model.predict(X_test, batch_size=128)

    return y_pred

def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score

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

def run_cv(X, y, labels):
    score_list = []
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2021)
    i = 0

    for train_idx, val_idx in kf.split(X, y):
        X_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ラベルをOne-Hotに変換
        y_train_onehot = to_categorical(y_train)
        y_val_onehot = to_categorical(y_val)

        print(X_train.shape)
        print(y_train_onehot.shape)

        model = build_model(len(labels))
        hist = learn_model(model, X_train, y_train_onehot, x_val, y_val_onehot)

        score = evaluate_model(model, x_val, y_val_onehot)
        y_pred = predict_model(model, x_val)
        y_pred = np.argmax(y_pred, axis=1)

        plot_evaluation(hist.history, 'loss', 'val_loss', 'loss', f'../figure/loss_{i}')
        plot_evaluation(hist.history, 'accuracy', 'val_accuracy', 'accuracy', f'../figure/acc_{i}')

        score_list.append(score[1])
        print(classification_report(y_val, y_pred, target_names=labels))
        cmx = confusion_matrix(y_val, y_pred)
        print(cmx)
        plot_cmx_heatmap(cmx, labels, f'../figure/cmx_{i}')
        i += 1
    
    print(score_list)
    print(np.array(score_list).mean())
# %%
# ポケモン
pokemon = load_npy_image('D:/OpenData/pokemon_dataset/Pokemon-Images-Dataset/npy/pokemon_sinnoh_128.npy')
pokemon.shape

# %%
# 花
daisy = load_npy_image('D:/OpenData/flowers/npy/daisy_128.npy')
dandelion = load_npy_image('D:/OpenData/flowers/npy/dandelion_128.npy')
rose = load_npy_image('D:/OpenData/flowers/npy/rose_128.npy')
sunflower = load_npy_image('D:/OpenData/flowers/npy/sunflower_128.npy')
tulip = load_npy_image('D:/OpenData/flowers/npy/tulip_128.npy')
print(daisy.shape)
print(dandelion.shape)
print(rose.shape)
print(sunflower.shape)
print(tulip.shape)
# %%
# 果物と野菜
fruit_vegetable = load_npy_image('D:/OpenData/Fruit-and-Vegetable-Image-Recognition/npy/fruit_vegetable_128.npy')
fruit_vegetable.shape

# %%
# ユニークラベル
flowers_labels = [re.split('[\\\.]',path)[-1] for path in gb.glob('D:/OpenData/flowers/raw/*')]
flowers_labels
# %%
fruit_vegetable_labels = [re.split('[\\\.]',path)[-1] for path in gb.glob('D:/OpenData/Fruit-and-Vegetable-Image-Recognition/train/*')]
fruit_vegetable_labels
# %%
labels = flowers_labels + fruit_vegetable_labels
print(len(labels))
labels

# %%
# データ準備
X, y = make_dataset(daisy, dandelion, rose, sunflower, tulip, fruit_vegetable, fruit_vegetable_labels)
print(X.shape)
print(y.shape)
# %%
# %%time
# run_cv(X, y, labels)


# %%
joblib.dump(model, 'model/base.model', compress=True)

# %%
print(device_lib.list_local_devices())