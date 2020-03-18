from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 2020
digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.8, test_size=0.2, random_state=SEED)

tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot.fit(X_train, y_train)

tpot.export('tpot_mnist_pipeline.py')


df = pd.DataFrame({
    'cust_id': [1, 2, 3, 1, 2, 3],
    'rating': [3, 5, 5, 4, 4, 4],
    'movie_id': [1, 1, 1, 2, 2, 2]
})

df_new = df.pivot(index='movie_id', columns='cust_id')
df_new.columns = df_new.columns.droplevel(0)
