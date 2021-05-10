import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from furupy.models import ModelLGB
from furupy.models import ModelRF
from furupy.models import ModelLR
from furupy.models import ModelKNN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import lightgbm as lgb


def make_train_test_data(path, features):
    df = pd.read_csv(path)
    df = df.loc[df['status']=='linetrace']

    tr_x = df.loc[df['flight_times']<=20, features]
    tr_y = df.loc[df['flight_times']<=20, 'state_num'].values
    te_x = df.loc[df['flight_times']>20, features]
    te_y = df.loc[df['flight_times']>20, 'state_num'].values

    return tr_x, tr_y, te_x, te_y


# 部分時系列を作成
def generate_part_seq(data_list, window):
    part_seq_list = []
    for i in range(len(data_list)-window+1):
        part_seq_list.append(data_list[i:i+window])
    
    return part_seq_list


# LightGBM
def rub_lgb(tr_x, tr_y, te_x, te_y, run_fold_name, params, load_path=None):
    lgbm = ModelLGB(run_fold_name, params)
    # 学習
    if load_path is None:
        build_lgb(tr_x, tr_y, lgbm)
    else:
        lgbm.load_model()
    
    # 予測
    pred = predict_lgb(te_x, te_y, lgbm)

    # 重要度可視化
    plot_lgb_importance(lgbm)

    # 評価
    print(classification_report(te_y, pred, target_names=['f', 'fccw', 'fcw']))
    print(confusion_matrix(te_y, pred))


def build_lgb(tr_x, tr_y, lgbm, issave=False):
    lgbm.train(tr_x, tr_y)
    if issave:
        lgbm.save_model()
        print('saved model')


def predict_lgb(te_x, te_y, lgbm):
    pred = lgbm.predict(te_x)
    pred = np.argmax(pred, axis=1)

    return pred


# 特徴量の重要度を確認
def plot_lgb_importance(lgbm):
    lgb.plot_importance(lgbm.model, height = 0.5, figsize = (4,8))
    plt.show()


# RandomForest
def run_rf(df):
    pass


# LogisticRegression
def run_lr(df):
    pass


# k-NearestNeighbors
def run_knn(df):
    pass


def main():
    features = ['tof', 'ord_yaw',
       'dif_roll_rad', 'dif_pitch_rad', 'dif_yaw_rad', 'vg_pitch_rad',
       'vg_roll_rad', 'vg_yaw_rad', 'vg_roll_rad_mean10',
       'vg_pitch_rad_mean10', 'vg_yaw_rad_mean10', 'agx_ms2', 'agy_ms2',
       'agz_ms2', 'agx_ms2_mean20', 'agy_ms2_mean20', 'agz_ms2_mean20',
       'vg_roll_rad_mean10_mean20', 'vg_pitch_rad_mean10_mean20',
       'vg_yaw_rad_mean10_mean20', 'agx_ms2_std20', 'agy_ms2_std20',
       'agz_ms2_std20', 'vg_roll_rad_mean10_std20',
       'vg_pitch_rad_mean10_std20', 'vg_yaw_rad_mean10_std20',
       'agx_ms2_kurt20', 'agy_ms2_kurt20', 'agz_ms2_kurt20',
       'vg_roll_rad_mean10_kurt20', 'vg_pitch_rad_mean10_kurt20',
       'vg_yaw_rad_mean10_kurt20', 'agx_ms2_skew20', 'agy_ms2_skew20',
       'agz_ms2_skew20', 'vg_roll_rad_mean10_skew20',
       'vg_pitch_rad_mean10_skew20', 'vg_yaw_rad_mean10_skew20',
       'agx_ms2_min20', 'agy_ms2_min20', 'agz_ms2_min20',
       'vg_roll_rad_mean10_min20', 'vg_pitch_rad_mean10_min20',
       'vg_yaw_rad_mean10_min20', 'agx_ms2_max20', 'agy_ms2_max20',
       'agz_ms2_max20', 'vg_roll_rad_mean10_max20',
       'vg_pitch_rad_mean10_max20', 'vg_yaw_rad_mean10_max20',
       'agx_ms2_median20', 'agy_ms2_median20', 'agz_ms2_median20',
       'vg_roll_rad_mean10_median20', 'vg_pitch_rad_mean10_median20',
       'vg_yaw_rad_mean10_median20', 'agx_ms2_rms20', 'agy_ms2_rms20',
       'agz_ms2_rms20', 'vg_roll_rad_mean10_rms20',
       'vg_pitch_rad_mean10_rms20', 'vg_yaw_rad_mean10_rms20']

    normal_path = '../data/processed/SE/2021-03-31_point-in-time.csv'
    # anomaly_path = '../data/processed/SE/2020-12-21_point-in-time_wind.csv'
    
    # データ取得
    train_x, train_y, test_x, test_y = make_train_test_data(normal_path, features)

    # LightGBM
    run_fold_name = 'lgb_1'
    params = {
        'num_leaves' : 31,
        'learning_rate' : 0.1,
        'metric': 'multi_logloss',
        'objective': 'multiclass',
        'num_class' : 5
    }
    rub_lgb(train_x, train_y, test_x, test_y, run_fold_name, params)



if __name__ == "__main__":
    main()