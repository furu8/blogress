from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

def make_datagen():
    datagen = ImageDataGenerator(
            # https://keras.io/ja/preprocessing/image/
            featurewise_center = False,            # データセット全体で，入力の平均を0にするかどうか
            samplewise_center = False,             # 各サンプルの平均を0にするかどうか
            featurewise_std_normalization = False, # 入力をデータセットの標準偏差で正規化するかどうか
            samplewise_std_normalization = False,  # 各入力をその標準偏差で正規化するかどうか
            zca_whitening = False,                 # ZCA白色化を適用するかどうか
            rotation_range = 30,                   # ランダムに±指定した角度の範囲で回転 
            width_shift_range = 0.3,               # ランダムに±指定した横幅に対する割合の範囲で左右方向移動
            height_shift_range = 0.3,              # ランダムに±指定した縦幅に対する割合の範囲で左右方向移動
            # zoom_range: 浮動小数点数または[lower，upper]．ランダムにズームする範囲．浮動小数点数が与えられた場合，[lower, upper] = [1-zoom_range, 1+zoom_range]です．
            horizontal_flip = True,                # 水平方向に入力をランダムに反転するかどうか
            vertical_flip = True,                  # 垂直方向に入力をランダムに反転するかどうか
            # validation_split = 0.2                 # 検証のために予約しておく画像の割合（厳密には0から1の間）
        )
    
    return datagen


def load_images(datagen, dir_path, subset):
    generator = datagen.flow_from_directory(
            # https://qiita.com/gal1996/items/00ed3589e13448496b4c
            train_dir=dir_path,
            # target_size=(224,224),
            batch_size=100,
            class_mode='binary',    # 複数のクラスであれば'categorical'、2クラスであれば'binary'
            # shuffle=True,
            # subset=subset           # validationかtrainのデータかを指定(make_datagen:validation_splitに関連)
        )
    
    return generator


def main():
    datagen = make_datagen()
    load_images(datagen, 'D:/Illust/Paimon/interim/paimon_face_only/', 'training')


if __name__ == "__main__":
    main()