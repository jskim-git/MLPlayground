import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

np.random.seed(2020)

# Forget Gate, Input Gate, Output Gate

df = pd.read_csv('./data/airline-passengers.csv')
df.set_index('Month')

dataset = pd.read_csv('./data/airline-passengers.csv', usecols=[1])
dataset = dataset.values
dataset = dataset.astype('float32')

sc = StandardScaler()
# sc = MinMaxScaler(feature_range=(0, 1))
dataset = sc.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


# print(train)

def create_dataset(dataset, lookback=1):
    X, y = [], []
    for i in range(len(dataset) - lookback - 1):
        a = dataset[i:(i + lookback), 0]
        X.append(a)
        y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(y)


lookback = 8
X_train, y_train = create_dataset(train, lookback)
X_test, y_test = create_dataset(test, lookback)

# Why reshape? => [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, lookback)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

y_train_pred = sc.inverse_transform(y_train_pred)
y_train = sc.inverse_transform(y_train)
y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test)

mse = mean_squared_error(y_train, y_train_pred[:, 0], squared=False)
print('Train RMSE: {:.2f}'.format(mse))
mse = mean_squared_error(y_test, y_pred[:, 0], squared=False)
print('Test RMSE: {:.2f}'.format(mse))

train_plot = np.empty_like(dataset)
train_plot[:, :] = np.nan
train_plot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

test_plot = np.empty_like(dataset)
test_plot[:, :] = np.nan
test_plot[len(y_train_pred)+(lookback*2)+1:len(dataset)-1, :] = y_pred

plt.plot(sc.inverse_transform(dataset))
plt.plot(train_plot)
plt.plot(test_plot)
plt.show()
