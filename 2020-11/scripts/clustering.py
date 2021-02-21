
import glob as gb
import shutil
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def load_image(path):
    image_list = []
    npy_image_list = []
    image_path_list = gb.glob(path)

    for i, image in enumerate(image_path_list):
        image_path = image.replace(chr(92), '/') # \を/に置換(windows特有)->macはchr(165)
        if i % 100 == 0: # 雑に進行状況出力
            print(i)
        
        # opencv版（PIL形式が必要なければこっちの方が少しだけ早い）
        img_npy = cv2.imread(image_path, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB) # RGB変換
        img_npy = cv2.resize(img_npy, (64, 64)) # リサイズ64x64

        # plt.imshow(img_npy)
        # plt.show()
        # break
        img_npy = img_npy.flatten() # 一次元化
        npy_image_list.append(img_npy/255) # 0~1に正規化

    return npy_image_list


def build_kmeans(cluster_num, df):
    model = KMeans(n_clusters=cluster_num)
    model.fit(df)

    return model


def plot_variance_ratio(pca):
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()


# 結果をクラスタごとにディレクトリに保存
def make_cluster_dir(load_path, save_path, model):
    # 保存先のディレクトリを空にして作成
    shutil.rmtree(save_path)
    os.mkdir(save_path)

    # クラスタごとのディレクトリ作成
    for i in range(model.n_clusters):
        cluster_dir = save_path + "cluster{}".format(i)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

    # 各ディレクトリにコピー保存
    image_path_list = gb.glob(load_path)
    for label, path in zip(model.labels_, image_path_list):
        shutil.copyfile(path, save_path + 'cluster{}/{}'.format(label, os.path.basename(path)))

def main():
    LOAD_PATH = 'D:/Illust/Paimon/interim/face_only/*'
    SAVE_PATH = 'D:/Illust/Paimon/interim/face_only_clustering/'
    # npy_image_list = load_image(LOAD_PATH)
    # df = pd.DataFrame(npy_image_list)
    # print(df.shape)
    
    # 主成分分析の実行
    # pca = PCA()
    # pca.fit(df)
    # feature = pca.transform(df)
    # print(feature.shape)
    # pca_df = pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(df))])
    # plot_variance_ratio(pca)
    # pca_df.to_csv('D:/Illust/Paimon/interim/face_only_pca.csv', index=False)

    pca_df = pd.read_csv('D:/Illust/Paimon/interim/face_only_pca.csv')
    train_df = pca_df.iloc[:, :1200]

    cluster_num = input('cluster_num >') 
    model = build_kmeans(cluster_num, train_df)
    # plt.scatter(pca_df['PC1'], pca_df['PC2'], c=model.labels_)
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=model.labels_)
    plt.show()
    make_cluster_dir(LOAD_PATH, SAVE_PATH, model)

    
if __name__ == "__main__":
    main()