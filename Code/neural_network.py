from pandas_datareader import data, wb
import pandas_datareader as pdr
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

symbols_list = []

stock =input('Enter the stock symbol: ')

symbols_list.append(stock)

d = {}
for ticker in symbols_list:
    d[ticker] = pdr.get_data_yahoo(ticker,'1980-01-01')
pan = pd.Panel(d)

df= pd.DataFrame(d[symbols_list[0]])

#print(df)

df=df[['Open','High','Low','Close']]

df['HL_PCT']=(df['High']-df['Close'])/df['Close']*100
df['PCT_change']=(df['Close']-df['Open'])/df['Open']*100

df=df[['Close','HL_PCT','PCT_change']]

forecast_col='Close'
df.fillna(-9999,inplace=True)

#forecast_out= int(math.ceil(0.01*len(df)))
forecast_out= 10
print ("The number of rows to be forcasted out:",forecast_out)

df['label']=df[forecast_col].shift(-forecast_out)
#print (df['label'])

#X: Features Y: Labels

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X=X[:-forecast_out]
#print (X)
X_lately = X[-forecast_out:]
#print (X_lately)

df.dropna(inplace=True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf = MLPRegressor(hidden_layer_sizes=100,solver='lbfgs')

clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)

print ("Achieved accuracy:",accuracy*100)

forecast_set = clf.predict(X_lately)

print ("The predicted ",forecast_out, "values are as follows:", forecast_set)