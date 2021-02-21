
import glob as gb
import shutil
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


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


def build_kmeans(df, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num)
    kmeans.fit(df)

    return kmeans


def build_pca(df):
    pca = PCA()
    pca.fit(df)

    return pca
    

def plot_contribution_rate(pca):
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()


def plot_scatter2d(df):
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='label', palette='bright', legend='full')
    plt.show()
    # plt.savefig('figure/pca_scatter2d.png')


def plot_scatter3d(df):
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    #.plotで描画
    for label in df['label'].values:
        ax.plot(df.loc[df['label']==label, 'PC1'], 
                df.loc[df['label']==label, 'PC2'], 
                df.loc[df['label']==label, 'PC3'], 
                alpha=0.8, marker=".", linestyle='None')
    plt.show()
    # plt.savefig('figure/pca_scatter3d.png')


# 結果をクラスタごとにディレクトリに保存
def make_cluster_dir(load_path, save_path, kmeans):
    # 保存先のディレクトリを空にして作成
    shutil.rmtree(save_path)
    os.mkdir(save_path)

    # クラスタごとのディレクトリ作成
    for i in range(kmeans.n_clusters):
        cluster_dir = save_path + "cluster{}".format(i)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir)

    # 各ディレクトリにコピー保存
    image_path_list = gb.glob(load_path)
    for label, path in zip(kmeans.labels_, image_path_list):
        shutil.copyfile(path, save_path + 'cluster{}/{}'.format(label, os.path.basename(path)))

    print('クラスタごとにファイル作成完了')


def main():
    LOAD_PATH = 'D:/Illust/Paimon/interim/face_only/*'
    SAVE_PATH = 'D:/Illust/Paimon/interim/face_only_clustering/'
    CSV_PATH = 'D:/Illust/Paimon/interim/face_only_pca.csv'       # 画像データを主成分分析した結果の保存先
    
    try:
        pca_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        # 画像読み込み
        npy_image_list = load_image(LOAD_PATH)
        df = pd.DataFrame(npy_image_list)
        print(df.shape)
        
        # 主成分分析の実行
        pca = build_pca(df)
        pca_df = pd.DataFrame(pca.transform(df), columns=["PC{}".format(x + 1) for x in range(len(df))])
        plot_contribution_rate(pca) # 累積寄与率可視化
        pca_df.to_csv(CSV_PATH, index=False) # 保存
    
    train_df = pca_df.iloc[:, :1200]
    
    cluster_num = int(input('cluster_num >')) # クラスタ数を入力
    kmeans = build_kmeans(train_df, cluster_num) # kmeansモデル構築
    make_cluster_dir(LOAD_PATH, SAVE_PATH, kmeans) # クラスタリング結果からディレクトリ作成

    pca_df['label'] = kmeans.labels_ 
    plot_scatter2d(pca_df)              # 二次元散布図
    plot_scatter3d(pca_df)              # 三次元散布図

    
if __name__ == "__main__":
    main()