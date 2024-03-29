import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
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

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


def load_image_npy(load_path, isdir=False):
    if isdir:
        return np.array([np.load(path, allow_pickle=True) for path in gb.glob(load_path)])
    else:
        return np.load(load_path, allow_pickle=True)


def make_train_test_data(image1, image2):
    X = np.concatenate([image1, image2])
    y = np.array([0] * len(image1) + [1] * len(image2)) # face:0, food:1
    
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

        # val_y = np.argmax(val_y, axis=1)
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