import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path):
    image_list = gb.glob(path)
    npy_image_list = []

    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB変換
        # plt.imshow(img)
        # plt.show()
        # break
        npy_image_list.append(img)

    return npy_image_list


def save_image_npy(read_path, save_path):
    image_list = load_image(read_path)
    np.save(save_path, image_list)


def main():
    FACE_LOAD_PATH = 'D:/Illust/Paimon/interim/face_only/*'
    FACE_SAVE_PATH = 'D:/Illust/Paimon/interim/npy_face_only/anime_face.npy'
    save_image_npy(FACE_LOAD_PATH, FACE_SAVE_PATH)
    
    FOOD_LOAD_PATH = 'D:/Illust/food-101/raw/images/*/*'
    FACE_SAVE_PATH = 'D:/Illust/food-101/interim/npy_food-101/food-101.npy'
    save_image_npy(FOOD_LOAD_PATH, FACE_SAVE_PATH)


if __name__ == "__main__":
    main()