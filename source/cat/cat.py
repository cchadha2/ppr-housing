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
                                                    test_size=0.33, 
                                                    random_state=SEED, 
                                                    shuffle=True
                                                    )

X_test, X_val, y_test, y_val = train_test_split(
                                                X_test,
                                                y_test, 
                                                test_size=0.5, 
                                                random_state=SEED, 
                                                shuffle=True
                                                )

train_pool = Pool(X_train.values, y_train.values, cat_features = [0,1,2,3,4,5,6])
test_pool = Pool(X_test.values, cat_features = [0,1,2,3,4,5,6])
val_pool = Pool(X_val.values, y_val.values, cat_features = [0,1,2,3,4,5,6])
   
print("Starting training. Train shape: {}".format(train_df.shape))

clf = CatBoostRegressor(iterations=1000, 
                        depth=3, 
                        learning_rate=0.8, 
                        loss_function='RMSE',
                        random_seed=SEED,
                        eval_metric='RMSE',
                        use_best_model=True
                        )

clf.fit(train_pool, eval_set=val_pool, early_stopping_rounds=10)

print("Predicting")
print("__________")
test_preds = clf.predict(test_pool)

print('RMSE: {}'.format(math.sqrt(mean_squared_error(y_test.values, test_preds))))
print('Price of my house: €{:.2f}'.format(np.asscalar(np.exp(clf.predict(np.array([['2019-01-04','Dublin 15','Dublin','No','Yes','Second-Hand Dwelling house /Apartment','greater than or equal to 38 sq metres and less than 125 sq metres']]))))))
