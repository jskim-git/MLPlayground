import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm

plt.rcParams.update({'figure.figsize': (10, 7),
                     'figure.dpi': 120})
# df = pd.read_csv('./data/airline-passengers.csv')
# df.set_index('Month')
df = pd.read_csv('./data/airline-passengers.csv', usecols=[1])

# Augmented Dickey-Fuller Test
result = adfuller(df)

print('ADF Statistic: {:.3f}'.format(result[0]))
print('p-value: {:5f}'.format(result[1]))

# Original Series
fig, axes = plt.subplots(3, 2, sharex='all')
axes[0, 0].plot(df.values)
axes[0, 0].set_title("Original Series")
plot_acf(df.values, ax=axes[0, 1], lags=130)

axes[1, 0].plot(np.diff(df.values, axis=0))
axes[1, 0].set_title("1st Order Diff")
plot_acf(np.diff(df.values, axis=0), ax=axes[1, 1], lags=130)

axes[2, 0].plot(np.diff(np.diff(df.values, axis=0), axis=0))
axes[2, 0].set_title("2nd Order Diff")
plot_acf(np.diff(np.diff(df.values, axis=0), axis=0), ax=axes[2, 1], lags=130)

# plt.show()

# model = ARIMA(df.values, order=(1, 2, 5))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())
#
# model_fit.plot_predict(dynamic=False)
# plt.show()

model = pm.auto_arima(df.values, start_p=1, start_q=1,
                      test='kpss', max_p=10, max_q=10,
                      m=12, seasonal=True, d=1,
                      start_P=0, D=1, trace=True,
                      error_action='ignore', suppress_warnings=True,
                      stepwise=True)
print(model.summary())
model.plot_diagnostics(figsize=(14, 10))
plt.show()

np.sort()



