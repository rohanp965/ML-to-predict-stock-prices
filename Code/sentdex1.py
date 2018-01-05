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

symbols_list = ['RELIANCE.NS']
d = {}
for ticker in symbols_list:
    d[ticker] = pdr.get_data_yahoo(ticker,'1980-12-01')
pan = pd.Panel(d)

df= pd.DataFrame(d[symbols_list[0]])

print(df)

df=df[['Open','High','Low','Close']]

df['HL_PCT']=(df['High']-df['Close'])/df['Close']*100
df['PCT_change']=(df['Close']-df['Open'])/df['Open']*100

df=df[['Close','HL_PCT','PCT_change']]

forecast_col='Close'
df.fillna(-9999,inplace=True)

forecast_out= int(math.ceil(0.01*len(df)))
print (forecast_out)

df['label']=df[forecast_col].shift(-forecast_out)


#X: Features Y: Labels

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X=X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)

clf=LinearRegression()

clf.fit(X_train,y_train)

accuracy=clf.score(X_test,y_test)

print (accuracy)


#Dunping classifier for further use
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)

 
accuracy = clf.score(X_test,y_test)

#print accuracy

forecast_set = clf.predict(X_lately)

print (forecast_set,accuracy,forecast_out)

df['Forecast'] = np.nan

last_date=df.iloc[-1].name
last_unix=last_date.microsecond/1000
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()