import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def make_image_df(path, label):
    df = pd.DataFrame(columns=['image_vec', 'label'])
    image_list = gb.glob(path)
    image_vec = []

    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        # plt.imshow(face_image)
        # plt.show()
        # break
        image_vec.append(img)

    df['image_vec'] = image_vec
    df['label'] = label

    return df

def main():
    # face_df = make_image_df('D:/Illust/Paimon/interim/face_only/*', 'face')
    # print(face_df.shape)
    food_train = tfds.load(name='food101', split='train')
    food_val = tfds.load(name='food101', split='validation')
    print(food_train.shape)

if __name__ == "__main__":
    main()