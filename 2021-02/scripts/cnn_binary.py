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
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model

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
        
        plot_evaluation(hist.history, 'loss', 'val_loss', 'loss')
        plot_evaluation(hist.history, 'accuracy', 'val_accuracy', 'accuracy')

        score = evaluate_model(model, X_test, y_test)
        y_pred = predict_model(model, X_test)

        # y_test = np.argmax(y_test, axis=1)
        y_pred = [1 if y > 0.9 else 0 for y in y_pred.flatten()]

        score_list.append(f1_score(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['food', 'face']))
    
    print(score_list)
    print(np.array(score_list).mean())


if __name__ == "__main__":
    main()