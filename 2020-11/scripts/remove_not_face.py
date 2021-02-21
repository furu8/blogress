import glob as gb
import shutil
import cv2
import os
import numpy as np
import pandas as pd


def main():
    LOAD_PATH1 = 'D:/Illust/Paimon/interim/face_only/*'
    LOAD_PATH2 = 'D:/Illust/Paimon/interim/face_only_judge_not_face/*'

    npy_images1 = np.array([os.path.basename(f) for f in gb.glob(LOAD_PATH1)])
    npy_images2 = np.array([os.path.basename(f) for f in gb.glob(LOAD_PATH2)])

    filenames = np.intersect1d(npy_images1, npy_images2) # 共通部分のファイル名を取得

    for filename in filenames:
        print("remove：{0}".format(filename))
        os.remove('D:/Illust/Paimon/interim/face_only/' + filename)


if __name__ == "__main__":
    main()