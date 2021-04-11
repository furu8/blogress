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
from tensorflow.keras.utils import to_categorical
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
    '''
    初期化 (initializer)
    Glorotの初期化法:sigmoid関数やtanh関数
    Heの初期化法:ReLU関数
    '''
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


def build_imagenet():
    # https://github.com/stratospark/food-101-keras
    # https://note.com/matsukoutennis/n/nfaa6b86ddf15
    # https://qiita.com/koshian2/items/b2d9c03ece95cf5f280a
    base_model = InceptionV3(weights='imagenet', 
                            include_top=False, 
                            input_tensor=Input(shape=(128, 128, 3)))
    x = base_model.output
    # x = AveragePooling2D(pool_size=(8, 8))(x)
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


def learn_model_generator(model, X_train, y_train, X_val, y_val, tr_datagen, va_datagen, path=None):
    early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=1.0e-3, 
                                patience=20, 
                                verbose=1)
    if path is None:
        hist = model.fit_generator(tr_datagen.flow(X_train, y_train, batch_size=32), 
                                    verbose=2, 
                                    steps_per_epoch=X_train.shape[0] // 32,
                                    epochs=100, 
                                    validation_data=(X_val, y_val), 
                                    callbacks=[early_stopping]
                                    )
    else:
        train_generator = make_dir_generator(tr_datagen, path)
        valid_generator = make_dir_generator(va_datagen, path)
        hist = model.fit_generator(train_generator,
                                verbose=2, 
                                steps_per_epoch=X_train.shape[0] // 1000,
                                epochs=100,
                                validation_data=valid_generator,
                                # callbacks=[early_stopping]
                                )
                    
    return hist


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


# ジェネレータ（学習、検証、テストデータが各ディレクトリに保存されているときに使う）
def make_dir_generator(datagen, path):
    generator = datagen.flow_from_directory(path,
                                            target_size=(64, 64),
                                            batch_size=1000,
                                            class_mode='categorical') # class_mode: binary/categorical

    return generator


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
    food_images = load_image_npy('D:/OpenData/food-101/interim/npy_food-101_64/npy_food-101_128.npy')/255

    # print(face_images[0])
    # print(food_images[0])

    resize_num = 128
    face_images = np.array([cv2.resize(face_image, (resize_num,resize_num)) for face_image in face_images])
    # food_images = np.array([cv2.resize(food_image, (resize_num,resize_num)) for food_image in food_images])

    print(face_images.shape)
    print(food_images.shape)
   
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
    # model = build_cnn_model()
    model = build_imagenet()

    hist = learn_model(model, X_train, y_train, X_val, y_val)
    # train_datagen = make_datagen()
    # valid_datagen = ImageDataGenerator()
    # hist = learn_model_generator(model, X_train, y_train, X_val, y_val, train_datagen, valid_datagen)
    
    save_file_name = 'v3.png'
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