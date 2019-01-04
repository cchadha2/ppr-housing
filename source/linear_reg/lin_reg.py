import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

train_df = pd.read_csv('output/processed_ppr_num.csv', encoding='latin-1')
SEED = 1001
le = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(
                                                    train_df,
                                                    train_df['Price'], 
                                                    test_size=0.2, 
                                                    random_state=SEED, 
                                                    shuffle=True
                                                    )

   
print("Starting training. Train shape: {}".format(train_df.shape))

clf = LinearRegression()

clf.fit(
    X_train.values,
    y_train.values,
)

print("Predicting")
print("__________")
test_preds = clf.predict(X_test.values)

print(r2_score(y_test, test_preds))
