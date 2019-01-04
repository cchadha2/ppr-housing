import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import r2_score

train_df = pd.read_csv('output/processed_ppr_cat.csv', encoding='latin-1')
SEED = 1001
le = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(
                                                    train_df,
                                                    train_df['Price'], 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    shuffle=True
                                                    )

train_pool = Pool(X_train.values, y_train.values, cat_features = [0,1,2,4,5,6,7])
test_pool = Pool(X_test.values, cat_features = [0,1,2,4,5,6,7])
   
print("Starting training. Train shape: {}".format(train_df.shape))

clf = CatBoostRegressor(iterations=100, 
                        depth=3, 
                        learning_rate=0.8, 
                        loss_function='RMSE')

clf.fit(train_pool)

print("Predicting")
print("__________")
test_preds = clf.predict(test_pool)

print(r2_score(y_test, test_preds))
