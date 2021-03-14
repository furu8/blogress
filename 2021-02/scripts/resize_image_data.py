import os
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


def load_image_npy(load_path, isdir=False):
    if isdir:
        return [np.load(path, allow_pickle=True) for path in gb.glob(load_path)]
    else:
        return np.load(load_path, allow_pickle=True)


def save_npy_image(save_path, images):
    np.save(save_path, images)
    print('save ', save_path)


face_images = load_image_npy('D:/Illust/Paimon/interim/npy_face_only/paimon_face_augmentation.npy')
food_images = load_image_npy('D:/OpenData/food-101/interim/npy_food-101.npy')

resize_num = 128
face_images = np.array([cv2.resize(face_image, (resize_num,resize_num)) for face_image in face_images])
food_images = np.array([cv2.resize(food_image, (resize_num,resize_num)) for food_image in food_images])

save_npy_image('D:/Illust/Paimon/interim/npy_face_only/paimon_face_augmentation_128.npy', face_images)
save_npy_image('D:/OpenData/food-101/interim/npy_food-101_128.npy', food_images)