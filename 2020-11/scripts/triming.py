import os
import cv2
import glob as gb

# 顔抽出
def get_face_img(image):
    # animeface
    cascade = cv2.CascadeClassifier('../config/lbpcascade_animeface.xml')
    
    face_image = cv2.imread(image, cv2.IMREAD_COLOR) # デフォルトカラー読み込み
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) # グレースケール化
    gray_image = cv2.equalizeHist(gray_image) # ヒストグラム平均化（見やすくなる）
    
    # face_list = cascade.detectMultiScale(gray_image, scaleFactor=1.01, minNeighbors=5, minSize=(24, 24)) # 見逃しを極力少なくパラメータ設定した場合
    face_list = cascade.detectMultiScale(gray_image, scaleFactor=1.09, minNeighbors=5, minSize=(24, 24)) # 誤検知を極力少なくパラメータ設定した場合
    
    return face_image, face_list

def save_face_image(save_path, face_img):
    if not os.path.exists(save_path): # DL済みの画像かどうか判定
        # # リサイズ
        # resize_face_img = resize_face_illust(face_img, 64)
        # 保存
        cv2.imwrite(save_path, face_img) 
        print(save_path + 'をトリミングし保存しました')

# def resize_face_illust(face_img, size):
#     width, height = size, size
#     resize_face_img = cv2.resize(face_img, (width, height))
    
#     return resize_face_img

# トリミング実行
def main():
    # トリミング対象の画像リスト
    image_list = gb.glob('D:/Illust/Paimon/raw/*')
    # print(len(image_list))
    
    for image in image_list:
        # 保存するファイル名
        save_file_name = str(os.path.basename(image)).split('.')[0]
       
        # 顔抽出
        try:
            face_image, face_list = get_face_img(image)
        except:
            print(save_file_name)
            print(image)
        
        if len(face_list) == 0:
            continue
        # キャラごと
        for i, face in enumerate(face_list):
            x, y, w, h = face # x始点、y始点、w幅、h高さ
            face_img = face_image[y:y+h, x:x+w] # トリミング
            # pngで保存
            save_path = 'D:/Illust/Paimon/interim/face/' + save_file_name + '_' +str(i) + '.png'
            save_face_image(save_path, face_img)


if __name__ == "__main__":
    main()