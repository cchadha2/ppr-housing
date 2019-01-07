import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

train_df = pd.read_csv('output/processed_ppr.csv', encoding='latin-1')
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

   
print("Starting training. Train shape: {}".format(train_df.shape))

clf = LinearRegression()

clf.fit(
    X_train,
    y_train,
)

print("Predicting")
print("__________")
test_preds = clf.predict(X_test)

print('RMSE: {}'.format(math.sqrt(mean_squared_error(y_test.values, test_preds))))
print('Price of my house: €{:.2f}'.format(np.asscalar(np.exp(clf.predict(np.array([[0,0,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]))))))
