# はじめにのはじめに

本記事で出てくる画像は一部、以下から引用してます。

> 株式会社MIHOYO
    [http://corp.mihoyo.co.jp/:embed:cite]

# はじめに

自分で集めた画像でCNNするために、TwitterAPIの検索機能を使って画像を集めています。  
集めている画像は特定の作品のキャラクターだったりするわけですが、CNNで分類モデルを作る上で画像にラベルを付けなくてはなりません。  
このラベル付けは言うまでもなく面倒で、時間がかかります。そこである程度自動化できないかと思ったわけです。  
したがって、今回はタイトルにある通り、pca（主成分分析）とkmeans（クラスタリング）を用いて画像のグルーピングを行いたいと思います。  

#  データ

キャラクターの画像は顔画像だけにあらかじめトリミングしてあるものを用います。  
その手法を知りたい方は以下を参照ください。  
古い記事ですが、lbpcascade_animeface.xmlはGoogleで検索した限りかなり広く使われています。  
検索すればいくらでも出てくるかと思いますので、オリジナルのものを今回は添付させていただきました。  

- ブログ記事

[http://ultraist.hatenablog.com/entry/20110718/1310965532:embed:cite]

- github

[https://github.com/nagadomi/lbpcascade_animeface:embed:cite]


# 問題

では、具体的に画像のラベル付けで何が面倒かを以下に挙げます。

1. キャラクターの顔認識精度に限界があり、誤検知が発生するため、顔画像以外の画像がデータ内に混在する

2. 特定のキャラクター1人のみラベルをつける場合、1枚1枚人が判定するのは時間がかかる

3. Twitterから集めているデータのため、ほぼ同じ画像がいくつか集まってしまう

## 1番の問題

lbpcascade_animeface.xmlは非常に良くできていますが、さすがに完璧な検知精度はありません。  
opencv側のパラメータをいじることで、ある程度は見落とさないよう検知したり過検知することは防げますが、それにも限界はあります。  
そのため、顔画像ではない画像が一つのクラスタとしてグルーピングされれば非常に手間が省けます。  
**結論を先に述べると、残念ながら本記事ではこの問題は解決できておりませんのでご注意ください**。

## 2番の問題

今回の場合、2021年2月現在、話題のゲーム「原神」のキャラクターの画像をTwitterから集めているわけですが、この中で筆者が欲しい画像はパイモンだけです。  
なぜパイモンの画像ばかり集めているかは別の記事で書くとして、パイモン以外の画像はすべて不要ということになります。  
クラスタリングをすることで、あるクラスタに一人もパイモンがいなければ、そのクラスタのディレクトリ名で保存されている画像は不必要な画像として扱えます。  
特定のキャラクターの画像を集める場合、こうすることで非常に効率が良くなることがわかるかと思います。

## 3番の問題
TwitterAPIの検索機能を使って、例えば「パイモン 原神」のように画像を集めるわけですが、どうしても同じ画像ばかり収集してしまうことがあります。  
これを問題視するかどうかはCNNの学習に関わるかと思いますが、削除したいモチベーションがあるなら、これもクラスタリングで解決できます。
あるクラスタに同じ画像が集まれば、その画像の内、一枚を除いて削除すれば画像の重複がなくなるからです。

 
#### ps
これらの問題はクラスタリングすることでラベル付けの手間を省くことに注力しているため、最終的には人の判断が介入します。



# プログラム

## 全体

```python

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


# 画像の読み込み
def load_image(path):
    image_list = []
    npy_image_list = []
    image_path_list = gb.glob(path)

    # 画像データを一枚ずつ読み込む
    for i, image in enumerate(image_path_list):
        image_path = image.replace(chr(92), '/') # \を/に置換(windows特有)->macはchr(165)
        if i % 100 == 0: # 雑に進行状況出力
            print(i)
        
        img_npy = cv2.imread(image_path, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
        img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB) # RGB変換
        img_npy = cv2.resize(img_npy, (64, 64)) # リサイズ64x64

        # plt.imshow(img_npy)
        # plt.show()
        # break
        img_npy = img_npy.flatten() # 一次元化
        npy_image_list.append(img_npy/255) # 0~1に正規化

    return npy_image_list


# kmeansのモデル構築
def build_kmeans(df, cluster_num):
    kmeans = KMeans(n_clusters=cluster_num, random_state=2021)
    kmeans.fit(df)

    return kmeans


# 主成分分析のモデル構築
def build_pca(df):
    pca = PCA()
    pca.fit(df)

    return pca
    

# 主成分分析の累積寄与率を可視化（この結果をもとに特徴ベクトルを決める）
def plot_contribution_rate(pca):
    fig = plt.figure()
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.show()
    # plt.savefig('../figure/pca_contribution_rate.png') 


# 主成分分析の第一主成分と第二主成分で散布図による可視化
def plot_scatter2d(df):
    fig = plt.figure()
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='label', palette='bright', legend='full')
    plt.show()
    # plt.savefig('../figure/pca_scatter2d.png')


# 主成分分析の第一主成分と第二主成分と第三主成分で散布図による可視化
def plot_scatter3d(df):
    # https://qiita.com/maskot1977/items/082557fcda78c4cdb41f
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
    # plt.savefig('../figure/pca_scatter3d.png')


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
    LOAD_PATH = 'D:/Illust/Paimon/interim/face_only/*'            # 画像データの読込先
    SAVE_PATH = 'D:/Illust/Paimon/interim/face_only_clustering/'  # 画像データをクラスタリングした結果の保存先
    CSV_PATH = 'D:/Illust/Paimon/interim/face_only_pca.csv'       # 画像データを主成分分析した結果の保存先
    
    try:
        # すでに画像データを主成分分析した結果のCSVファイルがあれば読み込む、なければexceptへ
        pca_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        # 画像読み込み
        npy_image_list = load_image(LOAD_PATH)
        df = pd.DataFrame(npy_image_list)
        print(df.shape)
        
        # 主成分分析の実行
        pca = build_pca(df)
        pca_df = pd.DataFrame(pca.transform(df), columns=["PC{}".format(x + 1) for x in range(len(df))])
        plot_contribution_rate(pca)          # 累積寄与率可視化
        pca_df.to_csv(CSV_PATH, index=False) # 保存
    
    # kmeansによるクラスタリング
    train_df = pca_df.iloc[:, :1200]               # 学習データ
    cluster_num = int(input('cluster_num >'))      # クラスタ数を入力
    kmeans = build_kmeans(train_df, cluster_num)   # kmeansモデル構築
    make_cluster_dir(LOAD_PATH, SAVE_PATH, kmeans) # クラスタリング結果からディレクトリ作成

    # 可視化
    pca_df['label'] = kmeans.labels_ 
    plot_scatter2d(pca_df)              # 二次元散布図
    plot_scatter3d(pca_df)              # 三次元散布図

    
if __name__ == "__main__":
    main()
```

## 補足説明

### 全体の流れ

1. 画像読み込み（一次元化済み）

2. 一次元化した画像を主成分分析にかけて、kmeans用の特徴ベクトルに

3. kmeansを実行してクラスタリング

4. クラスタリングした結果に基づいて、ディレクトリを作成し、各クラスタのディレクトリ名で画像をコピー保存

5. 最後に クラスタリング結果を可視化

### mainの例外処理

main関数の例外処理はCSV_PATHにあらかじめファイルを用意しない限り、原則except側に入ります。   
画像を1枚1枚読み込んで主成分分析するのには、そこそこ時間がかかります。  
何度もこの処理をするのは効率が悪いため、主成分分析した結果を一度実行したら保存しているわけです。二回目以降に実行時には保存した結果を読みに行きます。  
主成分分析した結果がクラスタリングするときの特徴ベクトルになります。

```python
try:
        # すでに画像データを主成分分析した結果のCSVファイルがあれば読み込む、なければexceptへ
        pca_df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        # 画像読み込み
        npy_image_list = load_image(LOAD_PATH)
        df = pd.DataFrame(npy_image_list)
        print(df.shape)

        # 主成分分析の実行
        pca = build_pca(df)
        pca_df = pd.DataFrame(pca.transform(df), columns=["PC{}".format(x + 1) for x in range(len(df))])
        plot_contribution_rate(pca)          # 累積寄与率可視化
        pca_df.to_csv(CSV_PATH, index=False) # 保存
```

### train_dfの1200
今回の場合、主成分分析した結果は約5000x5000の行列です。行が各主成分(つまり第5000主成分まである)、列が画像の枚数です。
主成分分析は次元圧縮のアルゴリズムで、5000ある特徴ベクトルを減らして（必要な情報だけに圧縮して）クラスタリング精度向上のために用います。
つまり、下記のコードで意味する1200は5000を1200まで減らしたということになります。  

```python
train_df = pca_df.iloc[:, :1200]               # 学習データ
```

1200にした理由は、主成分分析の累積寄与率というものを用います。plot_contribution_rate(pca)関数はそれを可視化したものになります。  
縦軸が1に近づけば近づくほど、その主成分だけでデータを説明できていることになります。  
1200で限りなく1に近いと判断し、今回は1200にしました。600でも、1800でも試す価値はあると思います（今回はやっていません）。  
なお、あらかじめ指定した累積寄与率を閾値として設定しておき、その値を超えたときの数を与えてやることもできます。

[f:id:Noleff:20210221230731p:plain]

### kmeansのクラスタ数
クラスタの数は標準入力で指定します。これをいくつにするかは正直適当です。  
今回は50にしました。

```python
cluster_num = int(input('cluster_num >'))      # クラスタ数を入力
kmeans = build_kmeans(train_df, cluster_num)   # kmeansモデル構築
```

参考までにクラスタリングした可視化結果を載せます。クラスタ数が50だと見えにくいので、クラスタ数を10で可視化しています。

- 二次元散布図

[f:id:Noleff:20210221231657p:plain]

- 三次元散布図

[f:id:Noleff:20210221231706p:plain]



# 結果

## ディレクトリ全体

[f:id:Noleff:20210221235048p:plain]

## 各ディレクトリ

### うまくいった例

#### 2番の問題を解決

このクラスタ数のディレクトリは特に触る必要がなくなります。
<figure class="figure-image figure-image-fotolife" title="パイモンしかいない">[f:id:Noleff:20210221233359p:plain]<figcaption>パイモンしかいない</figcaption></figure>

左上4番目にだけ凝光様がぽつんといます。白髪のため紛れ込んでしまったのでしょう。
<figure class="figure-image figure-image-fotolife" title="凝光様がいる">[f:id:Noleff:20210221233749p:plain]<figcaption>凝光様がいる</figcaption></figure>

今度は金髪のキャラクターが集まっているクラスタがありました。主人公がいっぱいいますね。一部クレーや凝光様もいるのが見てとれると思います（原神わからない方ごめんなさい）。    
なお、今回は髪色で同じクラスタになっているケースは他にも多くありました。    
<figure class="figure-image figure-image-fotolife" title="金髪キャラ">[f:id:Noleff:20210221234019p:plain]<figcaption>金髪キャラ</figcaption></figure>


#### 3番の問題を解決

ほぼ同じ画像のパイモンばかりしかいません。お好みで画像を削除できます。
<figure class="figure-image figure-image-fotolife" title="ほぼ同じパイモン">[f:id:Noleff:20210221233529p:plain]<figcaption>ほぼ同じパイモン1</figcaption></figure>
<figure class="figure-image figure-image-fotolife" title="はげパイモン草">[f:id:Noleff:20210221235426p:plain]<figcaption>ほぼ同じパイモン2</figcaption></figure>

### うまくいっていない例

しいて共通点を挙げるなら全体的に暗い画像となっている点です。しかし、明るい画像も一部含まれてもいます。  
このような場合、髪色のクラスタにグルーピングされず、結果として共通点のないクラスタになっていましまいた。  
今回うまくいっていないクラスタの大半がこのような暗めの画像でした。RGBだけでなくHSVの特徴ベクトルも必要かもしれませんね。  
パイモンがいなければディレクトリごと除去できますが、実は真ん中あたりにしれっといます。
<figure class="figure-image figure-image-fotolife" title="暗い画像？？">[f:id:Noleff:20210221234423p:plain]<figcaption>暗い画像？？</figcaption></figure>

さて、これは本当に共通点がないクラスタです。カラー画像や白黒画像が紛れ込んでます。  
同じ画像は同じクラスタには入っているようですが、それくらいしか共通点が見つかりませんでした。
<figure class="figure-image figure-image-fotolife" title="共通点不明">[f:id:Noleff:20210221234948p:plain]<figcaption>共通点不明</figcaption></figure>

# まとめ

今回はpcaとkmeansを用いて画像のグルーピングをしました。結果はそこそこ実用的ではありますが、まだまだ精度向上はできそうな気がします。  
今後の課題としては

- 1番の問題を解決する
- 画像の明るさを踏まえた特徴ベクトルを作成する

などがありますね。  
あとは別の次元圧縮アルゴリズムとクラスタリングアルゴリズムを使ってみると良いかもしれません。正直pcaとkmeansは王道ワンパターンなので（ただ王道を馬鹿にできないのもまた事実）。  
なお、最適なクラスタ数については考えません。kmeansでは正直不毛です。  
クラスタリングアルゴリズムではxmeans使うか階層型クラスタリングを使うか。  
次元圧縮アルゴリズムではt-SNEを使うか。  
といったところでしょうか。

# 参考文献

[http://corp.mihoyo.co.jp/:title]

[http://corp.mihoyo.co.jp/policy/guideline1.html:title]

[https://genshin.mihoyo.com/ja/game:title]

[https://rightcode.co.jp/blog/information-technology/machine-learning-image-clustering:title]

[https://qiita.com/maskot1977/items/082557fcda78c4cdb41f:title]

[https://blog.amedama.jp/entry/seaborn-plot:title]

[https://www.python.ambitious-engineer.com/archives/883:title]

[https://algorithm.joho.info/programming/python/matplotlib-scatter-plot-3d/:title]

[https://www.atmarkit.co.jp/ait/articles/1911/01/news026.html:title]

[https://note.nkmk.me/python-os-remove-rmdir-removedirs-shutil-rmtree/#:~:text=Python%E3%81%A7%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92%E5%89%8A%E9%99%A4,os.removedirs()%20%E3%82%82%E3%81%82%E3%82%8B%E3%80%82:title]

