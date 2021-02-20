import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img


def load_image(path):
    image_list = []
    npy_image_list = []
    image_path_list = gb.glob(path)

    for i, image in enumerate(image_path_list):
        image_path = image.replace(chr(92), '/') # \を/に置換(windows特有)->macはchr(165)
        if i % 100 == 0: # 雑に進行状況出力
            print(i)

        # opencv版（PIL形式が必要なけらばこっちの方が少しだけ早い）
        # img_npy = cv2.imread(image_path, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        # img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB) # RGB変換
        # img_npy = cv2.resize(img_npy, (64, 64)) # リサイズ64x64

        # keras版
        img = load_img(image_path, grayscale=False, color_mode='rgb', target_size=(64,64))
        img_npy = img_to_array(img) / 255

        # plt.imshow(img_npy)
        # plt.show()
        # break

        image_list.append(img)
        npy_image_list.append(img_npy)

    return image_list, npy_image_list


def make_datagen(rr=30, wsr=0.3, hsr=0.3, hf=True, vf=True):
    datagen = ImageDataGenerator(
            # https://keras.io/ja/preprocessing/image/
            featurewise_center = False,            # データセット全体で，入力の平均を0にするかどうか
            samplewise_center = False,             # 各サンプルの平均を0にするかどうか
            featurewise_std_normalization = False, # 入力をデータセットの標準偏差で正規化するかどうか
            samplewise_std_normalization = False,  # 各入力をその標準偏差で正規化するかどうか
            zca_whitening = False,                 # ZCA白色化を適用するかどうか
            rotation_range = rr,                   # ランダムに±指定した角度の範囲で回転 
            width_shift_range = wsr,               # ランダムに±指定した横幅に対する割合の範囲で左右方向移動
            height_shift_range = hsr,              # ランダムに±指定した縦幅に対する割合の範囲で左右方向移動
            # zoom_range: 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
            horizontal_flip = hf,                  # 水平方向に入力をランダムに反転するかどうか
            vertical_flip = vf,                    # 垂直方向に入力をランダムに反転するかどうか
            # validation_split = 0.2                 # 検証のために予約しておく画像の割合（厳密には0から1の間）
        )
    
    return datagen


def augument_image(images):
    image_list = []

    datagen = make_datagen()
    for image in images:
        image = image.reshape((1,) + image.shape)
        for img in datagen.flow(image, batch_size=1):
           image_list.append(img)
        break

    return image_list


def load_image(path):
    image_list = []
    npy_image_list = []
    image_path_list = gb.glob(path)

    for i, image in enumerate(image_path_list):
        image_path = image.replace(chr(92), '/') # \を/に置換(windows特有)->macはchr(165)
        if i % 100 == 0: # 雑に進行状況出力
            print(i)

        # opencv版（PIL形式が必要なけらばこっちの方が少しだけ早い）
        # img_npy = cv2.imread(image_path, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        # img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB) # RGB変換
        # img_npy = cv2.resize(img_npy, (64, 64)) # リサイズ64x64

        # keras版
        img = load_img(image_path, grayscale=False, color_mode='rgb', target_size=(64,64))
        img_npy = img_to_array(img) / 255

        # plt.imshow(img_npy)
        # plt.show()
        # break

        image_list.append(img)
        npy_image_list.append(img_npy)

    return image_list, npy_image_list


def save_npy_image(save_path, images):
    np.save(save_path, images)
    print('save ', save_path)


def save_image(save_path, load_path, images):
    filenames =  gb.glob(load_path)
    for filename, image in zip(filenames, images):
        filename = os.path.basename(filename)
        full_save_path = save_path + filename
        save_img(full_save_path, image)
    print('save ', full_save_path)


def main():
    # npy形式でロード＆セーブするときのパス
    FACE_LOAD_PATH = 'D:/Illust/Paimon/interim/paimon_face_only/*'              # 顔画像のロードパス
    FACE_SAVE_PATH = 'D:/Illust/Paimon/interim/npy_face_only/paimon_face.npy'   # 顔画像のセーブパス
    FOOD_LOAD_PATH = 'D:/OpenData/food-101/raw/images/*/*'                      # 飯画像のロードパス
    FOOD_SAVE_PATH = 'D:/OpenData/food-101/interim/npy_food-101.npy'            # 飯画像のセーブパス

    # 顔画像
    face_image_list, face_npy_image_list = load_image(FACE_LOAD_PATH)
    face_aug_image_list = augument_image(face_npy_image_list)
    # save_npy_image(FACE_SAVE_PATH, face_npy_image_list)
    save_image('D:/Illust/Paimon/interim/paimon_face_only_augmentation/', FACE_LOAD_PATH, face_aug_image_list)

    # 飯画像
    # food_image_list, food_npy_image_list = load_image(FOOD_LOAD_PATH)
    # save_npy_image(FOOD_SAVE_PATH, food_npy_image_list)


if __name__ == "__main__":
    main()