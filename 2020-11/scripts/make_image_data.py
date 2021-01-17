import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    image_list = gb.glob(path)
    npy_image_list = []

    for i, image in enumerate(image_list):
        if i % 100 == 0: # 雑に進行状況出力
            print(i)

        img = cv2.imread(image, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB変換
        img = cv2.resize(img, (64, 64)) # リサイズ64x64
        # plt.imshow(img)
        # plt.show()
        # break
        npy_image_list.append(img)

    return npy_image_list


def convert_img2npy(load_path, save_path, isdir=False):
    """
    isdir
        画像の保存先がディレクトリごとにわかれていかどうか
    """
    if isdir:
        for path in gb.glob(load_path):
            full_load_path = path + '/*'
            full_save_path = save_path + os.path.basename(path)
            image_list = load_image(full_load_path)
            np.save(full_save_path, image_list)
            print('save ', full_save_path)
    else:
        image_list = load_image(load_path)
        np.save(save_path, image_list)
        print('save ', save_path)


def main():
    FACE_LOAD_PATH = 'D:/Illust/Paimon/interim/face_only/*'                     # 顔画像のロードパス
    FACE_SAVE_PATH = 'D:/Illust/Paimon/interim/npy_face_only/anime_face.npy'    # 顔画像のセーブパス
    FOOD_LOAD_PATH = 'D:/Illust/food-101/raw/images/*/*'                        # 飯画像のロードパス
    FOOD_SAVE_PATH = 'D:/Illust/food-101/interim/npy_food-101/npy_food-101.npy' # 飯画像のセーブパス

    # 顔画像をnpy形式に変換
    convert_img2npy(FACE_LOAD_PATH, FACE_SAVE_PATH)

    # 飯画像をnpy形式に変換
    convert_img2npy(FOOD_LOAD_PATH, FOOD_SAVE_PATH)


if __name__ == "__main__":
    main()