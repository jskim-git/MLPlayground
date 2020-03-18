import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                 header=None, names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                                     'model', 'origin', 'car_name'], sep='\s+', na_values='?')

df = df.dropna().reset_index(drop=True)
df['horsepower'] = df['horsepower'].astype(float)

train_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

_ = df.hist(bins=10, figsize=(9, 7), grid=False)
plt.show()


