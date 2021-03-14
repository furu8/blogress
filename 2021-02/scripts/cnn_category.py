import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionV3

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


def load_image_npy(load_path, isdir=False):
    if isdir:
        return np.array([np.load(path, allow_pickle=True) for path in gb.glob(load_path)])
    else:
        return np.load(load_path, allow_pickle=True)


def make_train_test_data(image1, image2, labels):
    X = np.concatenate([image1, image2])

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
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(102, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def build_imagenet():
    print('build_imagenet')
    imagenet_model = InceptionV3(include_top=False, 
                                weights="imagenet", 
                                input_shape=(128,128,3))
    x = GlobalAveragePooling2D()(imagenet_model.layers[-1].output)
    x = Dense(102, activation="softmax")(x)

    # mixed4(132)から先を訓練する
    for i in range(133):
        imagenet_model.layers[i].trainable = False

    model = Model(imagenet_model.inputs, x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model


def learn_model(model, X_train, y_train, X_val, y_val):
    print('learn_model')

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
    y_pred = model.predict(X_test, batch_size=128)

    # y_pred = np.argmax(y_pred, axis=1)

    return y_pred


def save_model(path, model):
    model.save(path)
    print('saved model: ', path)


# 評価系のグラフをプロット
def plot_evaluation(eval_dict, key1, key2, ylabel):
    plt.plot(eval_dict[key1], label=key1)
    plt.plot(eval_dict[key2], label=key2)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def main():
    # GPUの動作確認
    # print(device_lib.list_local_devices())

    face_images = load_image_npy('D:/Illust/Paimon/interim/npy_face_only/paimon_face_augmentation.npy')
    # food_images = load_image_npy('D:/OpenData/food-101/interim/npy_food-101.npy')
    food_images = load_image_npy('D:/OpenData/food-101/interim/npy_food-101_64/*', isdir=True)
    
    print(face_images.shape)
    print(food_images.shape)

    labels = np.loadtxt('D:/OpenData/food-101/raw/meta/labels.txt', delimiter='\n', dtype=str)
    labels = labels[:len(gb.glob('D:/OpenData/food-101/interim/npy_food-101_64/*'))]
    print(labels.shape)

    X_train, X_test, y_train, y_test = make_train_test_data(face_images, food_images, labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=2021)

    # ラベルをOne-Hotに変換
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)

    model = build_cnn_model()
    # model = build_imagenet()
    hist = learn_model(model, X_train, y_train, X_val, y_val)
    
    plot_evaluation(hist.history, 'loss', 'val_loss', 'loss')
    plot_evaluation(hist.history, 'accuracy', 'val_accuracy', 'accuracy')

    score = evaluate_model(model, X_test, y_test)
    y_pred = predict_model(model, X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()