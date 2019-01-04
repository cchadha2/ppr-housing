import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
import math

train_df = pd.read_csv('output/processed_ppr_cat.csv', encoding='latin-1')
SEED = 1001

X = train_df.drop('Price', axis=1)
y = np.log(train_df['Price'])

X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    shuffle=True
                                                    )

train_pool = Pool(X_train.values, y_train.values, cat_features = [0,1,2,3,4,5,6])
test_pool = Pool(X_test.values, cat_features = [0,1,2,3,4,5,6])
   
print("Starting training. Train shape: {}".format(train_df.shape))

clf = CatBoostRegressor(iterations=100, 
                        depth=3, 
                        learning_rate=0.3, 
                        loss_function='RMSE')

clf.fit(train_pool)

print("Predicting")
print("__________")
test_preds = clf.predict(test_pool)

print(math.sqrt(mean_squared_error(y_test.values, test_preds)))