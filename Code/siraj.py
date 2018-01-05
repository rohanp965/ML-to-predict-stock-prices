# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:48:22 2017

@author: Dell
"""

from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm,time
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', 50, True)

model = Sequential()

model.add(LSTM(input_dim=1,output_dim=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))


model.add(Activation('linear'))

start=time.time()

model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

print (time.time()-start)


model.fit(X_train,y_train,batch_size=512,nb_epoch=1,validation_split=0.05)

predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

scores = model.evaluate(X_test,y_test,verbose=0)
print (scores)

plot_results_multiple(predictions,y_test,50)




