# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:38:44 2017

@author: Dell
"""
import numpy
import lstm
import time
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
import sys


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    canvas = FigureCanvasTkAgg(fig, master)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2TkAgg(canvas, master)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

#Main Run Thread
if __name__=='__main__':
    master = Tk()
    master.wm_title("LSTM Neural Network Predictor")
    L1 = Label(master, text="> Enter the stock symbol(YAHOO):",justify=LEFT)
    L1.pack()
    L1.config(anchor=NW)
    e = Entry(master)
    e.pack()
    e.focus_set()
    
    def callback():
        global_start_time = time.time()
        epochs  = 1
        seq_len = 50
        
        a=pdr.get_data_yahoo(e.get(),'1980-12-01')
        a.to_csv('a.csv',header=False,index=False,columns=['Close'])
        t.delete("1.0",END)
        t.insert(END,'> Loading data... \n')

        X_train, y_train, X_test, y_test = lstm.load_data('a.csv', seq_len, True)

        t.insert(END,'> Data Loaded. Compiling... \n')

        model = lstm.build_model([1, 50, 100, 1])

        model.fit(
	    X_train,
	    y_train,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05,
        )

        predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
        #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
        predicted = lstm.predict_point_by_point(model, X_test)   
        
        score = model.evaluate(X_test, y_test, batch_size=30,verbose=0)
        t.insert(END,'\n > LSTM test accuracy :'+str(score))
        t.insert(END,'\n > Training duration (s) :'+str(time.time() - global_start_time))
        plot_results(predicted, y_test)
    
    b = Button(master, text="Send", width=10, command=callback)
    b.pack()
    w = Canvas(master, width=100, height=20)
    w.pack()
    t = Text(master,height=10,width=80,borderwidth=3, relief="raised",background="lightgreen")
    t.tag_add('highlightline', '5.0', '6.0')
    t.pack()

    '''class TextRedirector(object):
        def __init__(self, widget, tag="stdout"):
            self.widget = widget
            self.tag = tag
            
        def write(self, str):
            self.widget.configure(state="normal")
            self.widget.insert("end", str, (self.tag,))
            self.widget.configure(state="disabled")
            
    sys.stdout = TextRedirector(t, "stdout")
    sys.stderr = TextRedirector(t, "stderr")'''
        
    scrollb = Scrollbar(master, command=t.yview)
    t['yscrollcommand'] = scrollb.set
    
    mainloop()

    
    #stock_name = input("> Enter the stock symbol(YAHOO):")
     
