# %%
import os
import pickle
import glob as gb
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array, array_to_img

# %%
###############################################################
# ポケモン

# シンホウ地方だけに抽出
def extract_sinnoh(pokemon_path_list):
    """
    387~493
    ナエトル~アルセウス
    """
    sinnoh_pokemon_path_list = []
    for pokemon in pokemon_path_list:
        # print(pokemon)

        # 数字だけにファイル名を頑張って除去
        number = int(re.sub(r'[f]', '', (pokemon.split('\\')[1].split('.')[0].split('-')[0])))
        
        # シンホウ地方
        if number >= 387 and number <= 493:
            # print(number)
            sinnoh_pokemon_path_list.append(pokemon)

    return sinnoh_pokemon_path_list


def load_images(image_path_list, ispokemon=False, isshow=False):
    npy_image_list = []

    for i, image in enumerate(image_path_list):
        image_path = image.replace(chr(92), '/') # \を/に置換(windows特有)->macはchr(165)
        if i % 100 == 0: # 雑に進行状況出力
            print(i)

        # 読み込み
        img = load_img(image_path, grayscale=False, color_mode='rgb', target_size=(128,128))
        img_npy = img_to_array(img)

        if ispokemon:
            index = np.where(img_npy[:, :, 2] == 0)
            img_npy[index] = [255, 255, 255] # 透過を白塗り

        img_npy = img_npy / 255 # 正規化

        if isshow:
            print(img_npy.shape)
            plt.imshow(img_npy)
            plt.show()
            break

        npy_image_list.append(img_npy)

    return npy_image_list

def save_npy_image(save_path, images):
    np.save(save_path, images)
    print('save ', save_path)

# %%
pokemon_path_list = gb.glob('D:/OpenData/pokemon_dataset/Pokemon-Images-Dataset/pokemon/*')
sinnoh_pokemon_path_list = extract_sinnoh(pokemon_path_list)
sinnoh_pokemon_path_list

# %%
# pokemon_image_list = load_images(sinnoh_pokemon_path_list, ispokemon=True, isshow=True)
pokemon_image_list = load_images(sinnoh_pokemon_path_list, ispokemon=True, isshow=False)
np.array(pokemon_image_list).shape

# %%
# 保存

# %%
###############################################################
# flowers
daisy_path_list = gb.glob('D:/OpenData/flowers/daisy/*')
dandelion_path_list = gb.glob('D:/OpenData/flowers/dandelion/*')
rose_path_list = gb.glob('D:/OpenData/flowers/rose/*')
sunflower_path_list = gb.glob('D:/OpenData/flowers/sunflower/*')
tulip_path_list = gb.glob('D:/OpenData/flowers/tulip/*')

# %%
daisy_image_list = load_images(daisy_path_list, isshow=True)
dandelion_image_list = load_images(dandelion_path_list, isshow=True)
rose_image_list = load_images(rose_path_list, isshow=True)
sunflower_image_list = load_images(sunflower_path_list, isshow=True)
tulip_image_list = load_images(tulip_path_list, isshow=True)

# %%
daisy_image_list = load_images(daisy_path_list)
dandelion_image_list = load_images(dandelion_path_list)
rose_image_list = load_images(rose_path_list)
sunflower_image_list = load_images(sunflower_path_list)
tulip_image_list = load_images(tulip_path_list)
print(np.array(daisy_image_list).shape)
print(np.array(dandelion_image_list).shape)
print(np.array(rose_image_list).shape)
print(np.array(sunflower_image_list).shape)
print(np.array(tulip_image_list).shape)

# %%
# 保存
# %%
###############################################################
# cifar-100-python <-低画質(32x32)なので断念

# from tensorflow.keras.datasets import cifar100
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# # https://kottas.hatenablog.com/entry/2019/02/04/000000
# # %%
# (train_images, train_labels), (test_images, test_labels) = cifar100.load_data(label_mode='coarse') # ＝全100クラス。'coarse'＝上位20個のスーパークラス
# train_images[2].shape
# # %%
# plt.imshow(train_images[2])
# plt.show()

# %%
###############################################################
# # なぜかファイル数が100枚になっていないので出力
# fruit_vegetable_path_list = gb.glob('D:/OpenData/Fruit-and-Vegetable-Image-Recognition/train/*') 
# fruit_vegetable_path_list

# for DIR in fruit_vegetable_path_list:
#     sum_file = sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR))
#     if sum_file != 100:
#         print(DIR)
#         print(sum_file)

# %%
fruit_vegetable_path_list = gb.glob('D:/OpenData/Fruit-and-Vegetable-Image-Recognition/train/*/*') 
fruit_vegetable_path_list
# %%
# fruit_vegetable_image_list = load_images(fruit_vegetable_path_list, isshow=True)
fruit_vegetable_image_list = load_images(fruit_vegetable_path_list, isshow=False)
np.array(fruit_vegetable_image_list).shape
# %%
