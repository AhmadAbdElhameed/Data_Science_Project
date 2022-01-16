## import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

## load dataset
df = pd.read_csv("Zomato_df.csv")
df.drop('Unnamed: 0',axis=1,inplace=True)

## prepare train and test data
X = df.drop("rate",axis = 1)
y = df['rate']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 10)

## Prape ExtraTreesRegressor Model
extra_model = ExtraTreesRegressor(n_estimators = 120)
extra_model.fit(X_train,y_train)
y_pred = extra_model.predict(X_test)

import pickle
## saving model to disk
pickle.dump(extra_model,open("extra_model.pkl","wb"))
model = pickle.load(open("extra_model.pkl","rb"))
print(y_pred)









