# %%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

import warnings
warnings.simplefilter('ignore')
#TensorFlowがGPUを認識しているか確認
from tensorflow.python.client import device_lib
# %%
df = pd.read_csv('qtdbsel102.txt', header=None, delimiter='\t')

print(df.shape)
# %%
ecg = df.iloc[:,2].values
ecg = ecg.reshape(len(ecg), -1)
print("length of ECG data:", len(ecg))
# %%
scaler = StandardScaler()
std_ecg = scaler.fit_transform(ecg)
# %%
plt.figure()

plt.style.use('ggplot')
plt.figure(figsize=(15,5))
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.plot(np.arange(45000), std_ecg[:45000], color='b')
plt.legend()

plt.show()
# %%
plt.figure()

plt.style.use('ggplot')
plt.figure(figsize=(15,5))
plt.xlabel('time')
plt.ylabel('ECG\'s value')
plt.plot(np.arange(5000), std_ecg[:5000], color='b')
plt.ylim(-3, 3)
x = np.arange(4200,4400)
y1 = [-3]*len(x)
y2 = [3]*len(x)
plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
plt.legend()

plt.show()
# %%
normal_cycle = std_ecg[5000:]
# %%
def generator(data, lookback, delay, pred_length, min_index, max_index, shuffle=False,
              batch_size=100, step=1):
    if max_index is None:
        max_index = len(data) - delay - pred_length - 1 
    i = min_index + lookback 

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, 
                                    size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                               lookback//step,
                               data.shape[-1]))

        targets = np.zeros((len(rows), pred_length))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay : rows[j] + delay + pred_length].flatten()

        yield samples, targets
# %%
lookback = 10
pred_length = 3
step = 1
delay = 1
batch_size = 100

# 訓練ジェネレータ
train_gen = generator(normal_cycle, 
                     lookback=lookback,
                     pred_length=pred_length,
                     delay=delay,
                     min_index=0,
                     max_index=20000,
                     shuffle=True,
                     step=step,
                     batch_size=batch_size)

val_gen = generator(normal_cycle, 
                   lookback=lookback,
                   pred_length=pred_length,
                   delay=delay,
                   min_index=20001,
                   max_index=30000,
                   step=step,
                   batch_size=batch_size)


# 検証データセット全体を調べるためにval_genから抽出する時間刻みの数
val_steps = (30001 - 20001 -lookback) // batch_size

# %%
model = Sequential()
model.add(layers.LSTM(35, return_sequences = True, input_shape=(None,normal_cycle.shape[-1])))
model.add(layers.LSTM(35))
model.add(layers.Dense(pred_length))

model.compile(optimizer=RMSprop(), loss="mse")

# %%
history = model.fit_generator(train_gen,
                              steps_per_epoch=200,
                              epochs=60,
                              validation_data=val_gen,
                              validation_steps=val_steps
                              )
# %%
