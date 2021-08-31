# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import math
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore')
#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib


# %%
# Timeから時間の差を追加
def differ_time(df):
    # Timeから時間の差を割り当てる
    df['dif_sec'] = df['time'].diff().fillna(0)
    df['cum_sec'] = df['dif_sec'].cumsum()
    return df

# 正規化
def act_minxmax_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    # 正規化したデータを新規のデータフレームに
    # df_mc = pd.DataFrame(scaler.transform(df), columns=df.columns) 
    # 正規化したデータをnumpyリストに
    mc_list = scaler.transform(df)

    return mc_list, scaler

# データ準備
def split_part_recurrent_data(data_list, look_back):
    X, Y = [], []
    for i in range(len(data_list)-look_back-1):
        X.append(data_list[i:(i+look_back), 0])
        Y.append(data_list[i + look_back, 0])
    
    return np.array(X), np.array(Y)

def create_lstm(look_back):
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1))) # input_shape=(系列長T, x_tの次元), output_shape=(units,)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

# LSTM学習
def act_lstm(model, train_x, train_y, batch_size, epochs):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, 
                                                        test_size=0.25, 
                                                        shuffle=False, 
                                                        random_state=2021)

    early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=1.0e-3, 
                                    patience=20, 
                                    verbose=1)

    history = model.fit(train_x, train_y, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=2,
            validation_data=(val_x, val_y), 
            callbacks=[early_stopping]
    )

    return model, history

# 評価系のグラフをプロット
def plot_evaluation(eval_dict, col1, col2, xlabel, ylabel):
    plt.plot(eval_dict[col1], label=col1)
    plt.plot(eval_dict[col2], label=col2)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

# %%
df = pd.read_csv('ecg.csv')

# 使うデータだけ抽出
test_df = df.iloc[:5000, :] 
train_df = df.iloc[5000:, :]

display(test_df)
display(train_df)

# %%
# 時間カラム追加
train_df = differ_time(train_df)
test_df = differ_time(test_df)

display(test_df)
display(train_df)

# %%
plt.figure(figsize=(16,4))
plt.plot(train_df['signal2'], label='Train')
plt.plot(test_df['signal2'], label='Test')
plt.legend()
plt.show()
# %%
plt.figure(figsize=(16,4))
plt.plot(train_df['cum_sec'], train_df['signal2'], label='Train')
plt.show()

plt.figure(figsize=(16,4))
plt.plot(test_df['cum_sec'], test_df['signal2'], color='tab:orange', label='Test')
plt.show()
# %%
anomaly_df = test_df[(test_df['cum_sec']>15)&(test_df['cum_sec']<20)]
plt.figure(figsize=(16,4))
plt.plot(anomaly_df['cum_sec'], anomaly_df['signal2'], color='tab:orange', label='Test')

anomaly_df = test_df[(test_df['cum_sec']>17)&(test_df['cum_sec']<17.5)] 
plt.plot(anomaly_df['cum_sec'], anomaly_df['signal2'], color='tab:red', label='Test')
plt.xlabel('sec')
# plt.show()

# %%
# ラベリング
test_df.loc[(test_df['cum_sec']>17)&(test_df['cum_sec']<17.5), 'label'] = 1
test_df['label'] = test_df['label'].fillna(0)
test_df[test_df['label']==1]
# %%
# 正規化
train_vec, train_scaler = act_minxmax_scaler(train_df[['signal2']])
test_vec, test_scaler = act_minxmax_scaler(test_df[['signal2']])

train_vec.shape

# %%
look_back = 250

train_x, train_y = split_part_recurrent_data(train_vec, look_back)
test_x, test_y = split_part_recurrent_data(test_vec, look_back)
display(pd.DataFrame(train_x))
display(pd.DataFrame(train_y))

# %%
print(train_x.shape)
print(test_x.shape)

# [samples, time steps, features]へ変形
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

print(train_x.shape)
print(test_x.shape)
# %%
# モデル作成
model = create_lstm(look_back)
model
# %%
# GPUの動作確認
# print(device_lib.list_local_devices())
batch_size = 100
epochs = 100

# モデル実行
model, hist = act_lstm(model, train_x, train_y, batch_size, epochs)

# %%
# テストデータに対する予測（評価のため訓練データも）
train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

# %%
# 正規化を元に戻す
trainPredict = train_scaler.inverse_transform(train_pred)
trainY = train_scaler.inverse_transform([train_y])
testPredict = test_scaler.inverse_transform(test_pred)
testY = test_scaler.inverse_transform([test_y])

# 平均二乗誤差のルートで評価
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# %%
# 誤差の収束具合を描画
# val_lossよりlossが大きければ未学習を疑え
plot_evaluation(hist.history, 'loss', 'val_loss', 'epoch', 'loss')
plt.savefig('eval_loss.png')

# %%
plt.figure(figsize=(20,4))
plt.plot(testY[0])
plt.plot(testPredict)
plt.show()

# %%
plt.figure(figsize=(20,4))
plt.plot(test_y)
plt.plot(test_pred)
# plt.show()
plt.savefig('real_pred_plot.png')

# %%
# 予測誤差（異常スコア）の計算
dist = test_y - test_pred[:,0]
u_dist = pow(dist, 2)
u_dist = u_dist / np.max(u_dist)

# %%
plt.figure(figsize=(20,4))
plt.plot(test_y)
plt.plot(u_dist, color='tab:red')
plt.savefig('anomaly_score.png')
# %%
# 予測誤差（異常スコア）の計算
dist = testY[0] - testPredict.flatten()
u_dist = pow(dist, 2)
u_dist = u_dist / np.max(u_dist)
u_dist
# %%
plt.figure(figsize=(20,4))
plt.plot(u_dist)
# plt.plot(test_y)

# %%
plt.hist(u_dist)

# %%
def Mahalanobis_dist(x):
    mean = np.mean(x, axis=0)
    cov = np.cov(x.T)
    d = np.dot(x-mean, np.linalg.inv(cov))
    d = d * (x-mean)
    d = np.sqrt(np.sum(d, axis=1))
    d /= np.max(d)
    return d

# %%
data = np.stack([test_y, test_pred[:,0]])
data.T

# %%
m_dist = Mahalanobis_dist(data.T)
m_dist.shape

# %%
plt.figure(figsize=(20,4))
# plt.plot(u_dist)
plt.plot(m_dist)
plt.plot(test_y)


# %%
plt.hist(m_dist)

# %%
result_df = pd.DataFrame()
result_df['test_pred'] = test_pred[:,0]
result_df['test_y'] = test_y
result_df['u_dist'] = u_dist
result_df['mae'] = np.abs(dist)
result_df['label'] = test_df['label'][look_back+1:].values
result_df['cum_sec'] = test_df['cum_sec'][look_back+1:].values
result_df
# %%
plt.scatter(result_df.loc[result_df['label']==0, 'test_y'], result_df.loc[result_df['label']==0, 'test_pred'])
plt.scatter(result_df.loc[result_df['label']==1, 'test_y'], result_df.loc[result_df['label']==1, 'test_pred'])

# %%
anomaly_df = result_df[(result_df['cum_sec']>15)&(result_df['cum_sec']<20)]
plt.figure(figsize=(16,4))
plt.plot(anomaly_df['cum_sec'], anomaly_df['test_y'], color='tab:orange', label='Test')
plt.plot(anomaly_df['cum_sec'], anomaly_df['u_dist'], color='tab:blue', label='Test')

anomaly_df = result_df[(result_df['cum_sec']>17)&(result_df['cum_sec']<17.5)] 
plt.plot(anomaly_df['cum_sec'], anomaly_df['test_y'], color='tab:red', label='Test')
plt.xlabel('sec')

# %%
anomaly_df = result_df[(result_df['cum_sec']>17)&(result_df['cum_sec']<17.5)] 
plt.figure(figsize=(16,4))
plt.plot(anomaly_df['cum_sec'], anomaly_df['test_pred'])
plt.plot(anomaly_df['cum_sec'], anomaly_df['test_y'])
plt.plot(anomaly_df['cum_sec'], anomaly_df['mae'])
# %%
anomaly_df
# %%
