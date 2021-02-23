import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


def load_image_npy(load_path, isdir=False):
    if isdir:
        return [np.load(path, allow_pickle=True) for path in gb.glob(load_path)]
    else:
        return np.load(load_path, allow_pickle=True)


def make_train_test_data(image1, image2):
    X = np.concatenate([image1, image2])
    y = np.array([1] * len(image1) + [0] * len(image2)) # face:1, food:0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=2021)

    return X_train, X_test, y_train, y_test


# モデル構築
def build_cnn_model():
    model = Sequential()

    # 入力画像 64x64x3 (縦の画素数)x(横の画素数)x(チャンネル数)
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal', input_shape=(64, 64, 3)))  # 28x28x1 -> 24x24x16
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 24x24x16 -> 12x12x16
    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu',
                    kernel_initializer='he_normal'))  # 12x12x16 -> 8x8x64
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 8x8x64 -> 4x4x64

    model.add(Flatten())  # 4x4x64-> 1024
    model.add(Dense(1, activation='sigmoid'))  # 1024 -> 10

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

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto') # https://qiita.com/yukiB/items/f45f0f71bc9739830002
    model.fit(X_train, y_train, 
                batch_size=1000, 
                verbose=2, 
                epochs=10, 
                validation_data=(X_val, y_val), 
                callbacks=[early_stopping])
    
    return model


def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def predict_model(model, X_test):
    y_pred = model.predict(X_test, batch_size=128)

    # y_pred = np.argmax(y_pred, axis=1)

    return y_pred


def main():
    # GPUの動作確認
    # print(device_lib.list_local_devices())

    face_image = load_image_npy('D:/Illust/Paimon/interim/npy_face_only/paimon_face_augmentation.npy')
    food_image = load_image_npy('D:/OpenData/food-101/interim/npy_food-101.npy')
    food_image = food_image[np.random.choice(food_image.shape[0], 10000, replace=False), :] # food_image101000枚の画像からランダムに10000枚抽出

    X_train, X_test, y_train, y_test = make_train_test_data(face_image, food_image)
    
    print(face_image.shape)
    print(food_image.shape)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = build_cnn_model()
    model = learn_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    y_pred = predict_model(model, X_test)
    # y_test = np.argmax(y_test, axis=1)
    print(y_test)
    print(y_pred)
    print(y_test.flatten())
    print(y_pred.flatten().astype(np.int32))
    print(classification_report(y_test.flatten(), y_pred.flatten().astype(np.int32), target_names=['food', 'face']))


if __name__ == "__main__":
    main()