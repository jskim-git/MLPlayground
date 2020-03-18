import pandas as pd
import numpy as np

import warnings
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pylab import rcParams

warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = 18, 8

df = pd.read_excel('./data/Superstore.xls')
furniture = df.loc[df.Category == 'Furniture']

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City',
        'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity',
        'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')

furniture = furniture.set_index('Order Date')
y = furniture['Sales']

y_month = y.resample('MS').mean()

decomp = sm.tsa.seasonal_decompose(y_month, model='additive')
fig = decomp.plot()
plt.show()

p = q = range(0, 4)
d = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

res = []
aic = []
# Grid Search? similar
for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(y_month, order=param,
                                              seasonal_order=seasonal_param,
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            results = model.fit()
            res.append([param, seasonal_param, results.aic])
            print('ARIMA{}x{} - AIC:{}'.format(param, seasonal_param, results.aic))

        except:
            continue

print(res)
model = sm.tsa.statespace.SARIMAX(y_month, order=(0, 0, 0),
                                  seasonal_order=(2, 2, 0, 12))
results = model.fit()
print(results.summary().tables[1])
