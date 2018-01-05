#Used for numpy arrays
import numpy as np
#Used to read data from CSV file
import pandas as pd
#Used to convert date string to numerical value
from datetime import datetime, timedelta
#Used to plot data
import matplotlib.pyplot as mpl
#Used to scale data
from sklearn.preprocessing import StandardScaler
#Used to perform CV
from sklearn.cross_validation import KFold

#Gives a list of timestamps from the start date to the end date
#
#startDate:     The start date as a string xxxx-xx-xx 
#endDate:       The end date as a string year-month-day
#period:		'daily', 'weekly', or 'monthly'
#weekends:      True if weekends should be included; false otherwise
#return:        A numpy array of timestamps        
def DateRange(startDate, endDate, period, weekends = False):
    #The start and end date
    sd = datetime.strptime(startDate, '%Y-%m-%d')
    ed = datetime.strptime(endDate, '%Y-%m-%d')
    #Invalid start and end dates
    if(sd > ed):
        raise ValueError("The start date cannot be later than the end date.")
    #One time period is a day
    if(period == 'daily'):
        prd = timedelta(1)
    #One prediction per week
    elif(period == 'weekly'):
        prd = timedelta(7)
    #one prediction every 30 days ("month")
    else:
        prd = timedelta(30)
    #The final list of timestamp data
    dates = []
    cd = sd
    while(cd <= ed):
        #If weekdays are included or it's a weekday append the current ts
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            dates.append(cd.timestamp())
        #Onto the next period
        cd = cd + prd
    return np.array(dates)
    
#Given a date, returns the previous day
#
#startDate:     The start date as a datetime object
#weekends:      True if weekends should counted; false otherwise
def DatePrevDay(startDate, weekends = False):
    #One day
    day = timedelta(1)
    cd = datetime.fromtimestamp(startDate)
    while(True):
        cd = cd - day
        if(weekends or (cd.date().weekday() != 5 and cd.date().weekday() != 6)):
            return cd.timestamp()
    #Should never happen
    return None

#Load data from the CSV file. Note: Some systems are unable
#to give timestamps for dates before 1970. This function may
#fail on such systems.
#
#path:      The path to the file
#return:    A data frame with the parsed timestamps
def ParseData(path):
    #Read the csv file into a dataframe
    df = pd.read_csv(path)
    #Get the date strings from the date column
    dateStr = df['Date'].values
    D = np.zeros(dateStr.shape)
    #Convert all date strings to a numeric value
    for i, j in enumerate(dateStr):
        #Date strings are of the form year-month-day
        D[i] = datetime.strptime(j, '%Y-%m-%d').timestamp()
    #Add the newly parsed column to the dataframe
    df['Timestamp'] = D
    #Remove any unused columns (axis = 1 specifies fields are columns)
    return df.drop('Date', axis = 1) 
    
        
#Given dataframe from ParseData
#plot it to the screen
#
#df:        Dataframe returned from 
#p:         The position of the predicted data points
def PlotData(df, p = None):
    if(p is None):
        p = np.array([])
    #Timestamp data
    ts = df.Timestamp.values
    #Number of x tick marks
    nTicks= 10
    #Left most x value
    s = np.min(ts)
    #Right most x value
    e = np.max(ts)
    #Total range of x values
    r = e - s
    #Add some buffer on both sides
    s -= r / 5
    e += r / 5
    #These will be the tick locations on the x axis
    tickMarks = np.arange(s, e, (e - s) / nTicks)
    #Convert timestamps to strings
    strTs = [datetime.fromtimestamp(i).strftime('%m-%d-%y') for i in tickMarks]
    mpl.figure()
    #Plots of the high and low values for the day
    mpl.plot(ts, df.High.values, color = '#727272', linewidth = 1.618, label = 'Actual')
    #Predicted data was also provided
    if(len(p) > 0):
        mpl.plot(ts[p], df.High.values[p], color = '#7294AA', linewidth = 1.618, label = 'Predicted')
    #Set the tick marks
    mpl.xticks(tickMarks, strTs, rotation='vertical')
    #Set y-axis label
    mpl.ylabel('Stock High Value (USD)')
    #Add the label in the upper left
    mpl.legend(loc = 'upper left')
    mpl.show()

#A class that predicts stock prices based on historical stock data
class StockPredictor:
    
    #The (scaled) data frame
    D = None
    #Unscaled timestamp data
    DTS = None
    #The data matrix
    A = None
    #Target value matrix
    y = None
    #Corresponding columns for target values
    targCols = None
    #Number of previous days of data to use
    npd = 1
    #The regressor model
    R = None
    #Object to scale input data
    S = None
    
    #Constructor
    #nPrevDays:     The number of past days to include
    #               in a sample.
    #rmodel:        The regressor model to use (sklearn)
    #nPastDays:     The number of past days in each feature
    #scaler:        The scaler object used to scale the data (sklearn)
    def __init__(self, rmodel, nPastDays = 1, scaler = StandardScaler()):
        self.npd = nPastDays
        self.R = rmodel
        self.S = scaler
        
    #Extracts features from stock market data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of samples
    def _ExtractFeat(self, D):
        #One row per day of stock data
        m = D.shape[0]
        #Open, High, Low, and Close for past n days + timestamp and volume
        n = self._GetNumFeatures()
        B = np.zeros([m, n])
        #Preserve order of spreadsheet
        for i in range(m - 1, -1, -1):
            self._GetSample(B[i], i, D)
        #Return the internal numpy array
        return B
        
    #Extracts the target values from stock market data
    #
    #D:         A dataframe from ParseData
    #ret:       The data matrix of targets and the
    
    def _ExtractTarg(self, D):
        #Timestamp column is not predicted
        tmp = D.drop('Timestamp', axis = 1)
        #Return the internal numpy array
        return tmp.values, tmp.columns
        
    #Get the number of features in the data matrix
    #
    #n:         The number of previous days to include
    #           self.npd is  used if n is None
    #ret:       The number of features in the data matrix
    def _GetNumFeatures(self, n = None):
        if(n is None):
            n = self.npd
        return n * 7 + 1
        
    #Get the sample for a specific row in the dataframe. 
    #A sample consists of the current timestamp and the data from
    #the past n rows of the dataframe
    #
    #r:         The array to fill with data
    #i:         The index of the row for which to build a sample
    #df:        The dataframe to use
    #return;    r
    def _GetSample(self, r, i, df):
        #First value is the timestamp
        r[0] = df['Timestamp'].values[i]
        #The number of columns in df
        n = df.shape[1]
        #The last valid index
        lim = df.shape[0]
        #Each sample contains the past n days of stock data; for non-existing data
        #repeat last available sample
        #Format of row:
        #Timestamp Volume Open[i] High[i] ... Open[i-1] High[i-1]... etc
        for j in range(0, self.npd):
            #Subsequent rows contain older data in the spreadsheet
            ind = i + j + 1
            #If there is no older data, duplicate the oldest available values
            if(ind >= lim):
                ind = lim - 1
            #Add all columns from row[ind]
            for k, c in enumerate(df.columns):
                #+ 1 is needed as timestamp is at index 0
                r[k + 1 + n * j] = df[c].values[ind]
        return r
        
    #Attempts to learn the stock market data
    #given a dataframe taken from ParseData
    #
    #D:         A dataframe from ParseData
    def Learn(self, D):
        #Keep track of the currently learned data
        self.D = D.copy()
        #Keep track of old timestamps for indexing
        self.DTS = np.copy(D.Timestamp.values)
        #Scale the data
        self.D[self.D.columns] = self.S.fit_transform(self.D)
        #Get features from the data frame
        self.A = self._ExtractFeat(self.D)
        #Get the target values and their corresponding column names
        self.y, self.targCols = self._ExtractTarg(self.D)
        #Create the regressor model and fit it
        self.R.fit(self.A, self.y)
        
    #Predicts values for each row of the dataframe. Can be used to 
    #estimate performance of the model
    #
    #df:            The dataframe for which to make prediction
    #return:        A dataframe containing the predictions
    def PredictDF(self, df):
        #Make a local copy to prevent modifying df
        D = df.copy()
        #Scale the input data like the training data
        D[D.columns] = self.S.transform()
        #Get features
        A = self._ExtractFeat(D)
        #Construct a dataframe to contain the predictions
        #Column order was saved earlier 
        P = pd.DataFrame(index = range(A.shape[0]), columns = self.targCols)
        #Perform prediction 
        P[P.columns] = self.R.predict(A)
        #Add the timestamp (already scaled from above)
        P['Timestamp'] = D['Timestamp'].values
        #Scale the data back to original range
        P[P.columns] = self.S.inverse_transform(P)
        return P
        
    #Predict the stock price during a specified time
    #
    #startDate:     The start date as a string in yyyy-mm-dd format
    #endDate:       The end date as a string yyyy-mm-dd format
    #period:		'daily', 'weekly', or 'monthly' for the time period
    #				between predictions
    #return:        A dataframe containing the predictions or
    def PredictDate(self, startDate, endDate, period = 'weekly'):
        #Create the range of timestamps and reverse them
        ts = DateRange(startDate, endDate, period)[::-1]
        m = ts.shape[0]
        #Prediction is based on data prior to start date
        #Get timestamp of previous day
        prevts = DatePrevDay(ts[-1])
        #Test if there is enough data to continue
        try:
            ind = np.where(self.DTS == prevts)[0][0]
        except IndexError:
            return None
        #There is enough data to perform prediction; allocate new data frame
        P = pd.DataFrame(np.zeros([m, self.D.shape[1]]), index = range(m), columns = self.D.columns)
        #Add in the timestamp column so that it can be scaled properly
        P['Timestamp'] = ts
        #Scale the timestamp (other fields are 0)
        P[P.columns] = self.S.transform(P)
        #B is to be the data matrix of features
        B = np.zeros([1, self._GetNumFeatures()])
        #Add extra last entries for past existing data
        for i in range(self.npd):
            #If the current index does not exist, repeat the last valid data
            curInd = ind + i
            if(curInd >= self.D.shape[0]):
                curInd = curInd - 1
            #Copy over the past data (already scaled)
            P.loc[m + i] = self.D.loc[curInd]
        #Loop until end date is reached
        for i in range(m - 1, -1, -1):
            #Create one sample 
            self._GetSample(B[0], i, P)
            #Predict the row of the dataframe and save it
            pred = self.R.predict(B).ravel()
            #Fill in the remaining fields into the respective columns
            for j, k in zip(self.targCols, pred):
                P.set_value(i, j, k)
        #Discard extra rows needed for prediction
        P = P[0:m]
        #Scale the dataframe back to the original range
        P[P.columns] = self.S.inverse_transform(P)
        return P
        
    #Test the predictors performance and
    #displays results to the screen
    #
    #D:             The dataframe for which to make prediction
    def TestPerformance(self, df = None):
        #If no dataframe is provided, use the currently learned one
        if(df is None):
            D = self.D
        else:
            D = self.S.transform(df.copy())
        #Get features from the data frame
        A = self._ExtractFeat(D)
        #Get the target values and their corresponding column names
        y, _ = self._ExtractTarg(D)
        #Begin cross validation
        kf = KFold(A.shape[0])
        for trn, tst in kf:
            s1 = self.R.score(A, y)
            s2 = self.R.score(A[tst], y[tst])
            s3 = self.R.score(A[trn], y[trn])
            print('C-V:\t' + str(s1) + '\nTst:\t' + str(s2) + '\nTrn:\t' + str(s3))
            
            
import tensorflow as tf
import numpy as np

#Return the classification accuracy
#given a vector of target labels and
#predicted labels
#y: The target labels
#yHat: The predicted labels
#return: The percentage correct
def _Accuracy(y, yHat):
    n = float(len(y))
    return np.sum(y == yHat) / n

#Create the MLP variables for TF graph
#_X: The input matrix
#_W: The weight matrices
#_B: The bias vectors
#_AF: The activation function
def _CreateMLP(_X, _W, _B, _AF):
    n = len(_W)
    for i in range(n - 1):
        _X = _AF(tf.matmul(_X, _W[i]) + _B[i]) 
    return tf.matmul(_X, _W[n - 1]) + _B[n - 1]

#Add L2 regularizers for the weight and bias matrices
#_W: The weight matrices
#_B: The bias matrices
#return: tensorflow variable representing l2 regularization cost
def _CreateL2Reg(_W, _B):
    n = len(_W)
    regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
    for i in range(1, n):
        regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
    return regularizers

#Create weight and bias vectors for an MLP
#layers: The number of neurons in each layer (including input and output)
#return: A tuple of lists of the weight and bias matrices respectively 
def _CreateVars(layers):
    weight = []
    bias = []
    n = len(layers)
    for i in range(n - 1):
        #Fan-in for layer; used as standard dev
        lyrstd = np.sqrt(1.0 / layers[i])
        curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev = lyrstd))
        weight.append(curW)
        curB = tf.Variable(tf.random_normal([layers[i + 1]], stddev = lyrstd))
        bias.append(curB)
    return (weight, bias)   

#Helper function for selecting an activation function
#name: The name of the activation function
#return: A handle for the tensorflow activation function
def _GetActvFn(name):
    if name == 'tanh':
        return tf.tanh
    elif name == 'sig':
        return tf.sigmoid
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'relu6':
        return tf.nn.relu6
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'softsign':
        return tf.nn.softsign
    return None

#Helper function for getting a tensorflow optimizer
#name:    The name of the optimizer to use
#lr:      The learning rate if applicable
#return;  A the tensorflow optimization object
def _GetOptimizer(name, lr):
    if(name == 'adam'):
        return tf.train.AdamOptimizer(learning_rate = lr)
    elif(name == 'grad'):
        return tf.train.GradientDescentOptimizer(learning_rate = lr)
    elif(name == 'adagrad'):
        return tf.train.AdagradOptimizer(learning_rate = lr)
    elif(name == 'ftrl'):
        return tf.train.FtrlOptimizer(learning_rate = lr)
    return None

#Gives the next batch of samples of size self.batSz or the remaining
#samples if there are not that many
#A: Samples to choose from
#y: Targets to choose from
#cur: The next sample to use
#batSz: Size of the batch
#return: A tuple of the new samples and targets
def _NextBatch(A, y, cur, batSz):
    m = len(A)
    nxt = cur + batSz
    if(nxt > m):
        nxt = m
    return (A[cur:nxt], y[cur:nxt])

#Multi-Layer Perceptron for Classification
class MLPC:

    #Predicted outputs
    pred = None
    #The loss function
    loss = None
    #The optimization method
    optmzr = None
    #Max number of iterations
    mItr = None
    #Error tolerance
    tol = None
    #Tensorflow session
    sess = None
    #Input placeholder
    x = None
    #Output placeholder
    y = None
    #Boolean for toggling verbose output
    vrbse = None
    #Batch size
    batSz = None
    #The class labels
    _classes = None

    def __init__(self, layers, actvFn = 'tanh', optmzr = 'adam', learnRate = 0.001, decay = 0.9, 
                 maxItr = 2000, tol = 1e-2, batchSize = None, verbose = False, reg = 0.001):
        #Parameters
        self.tol = tol
        self.mItr = maxItr
        self.n = len(layers)
        self.vrbse = verbose
        self.batSz = batchSize
        #Input size
        self.x = tf.placeholder("float", [None, layers[0]])
        #Output size
        self.y = tf.placeholder("float", [None, layers[-1]])
        #Setup the weight and bias variables
        weight, bias = _CreateVars(layers)   
        #Create the tensorflow model
        self.pred = _CreateMLP(self.x, weight, bias, _GetActvFn(actvFn))
        #Cross entropy loss function
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        #Use regularization to prevent over-fitting
        if(reg is not None):
            self.loss += _CreateL2Reg(weight, bias) * reg
        self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
        #Initialize all variables on the TF session
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
    #Fit the MLP to the data
    #param A: numpy matrix where each row is a sample
    #param y: numpy matrix of target values
    def fit(self, A, y):
        m = len(A)
        y = self.to1Hot(y)
        #Begin training
        for i in range(self.mItr):
            #Batch mode or all at once
            if(self.batSz is None):
                self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
            else:
                for j in range(0, m, self.batSz):
                    batA, batY = _NextBatch(A, y, j, self.batSz)
                    self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
            err = np.sqrt(np.sum(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y})) / m)
            if(self.vrbse):
                print("Iter " + str(i + 1) + ": " + str(err))
            if(err < self.tol):
                break
    
    #Predict the output given the input (only run after calling fit)
    #param A: The input values for which to predict outputs 
    #return: The predicted output values (one row per input sample)
    def predict(self, A):
        if(self.sess == None):
            print("Error: MLPC has not yet been fitted.")
            return None
        #Get the predicted indices
        res = np.argmax(self.sess.run(self.pred, feed_dict={self.x:A}), 1)
        res.shape = [-1]
        #Return prediction using the original labels
        return np.array([self._classes[i] for i in res])
        
    #Predicts the ouputs for input A and then computes the classification error
    #The predicted values and the actualy values
    #param A: The input values for which to predict outputs 
    #param y: The actual target values
    #return: The percent of outputs predicted correctly
    def score(self, A, y):
        yHat = self.predict(A)
        return _Accuracy(y, yHat)

    #Creates an array of 1-hot vectors
    #based on a vector of class labels
    #y: The vector of class labels
    #return: The 1-Hot encoding of y
    def to1Hot(self, y):
        lbls = list(set(list(y)))
        lbls.sort()
        lblDic = {}
        self._classes = []
        for i in range(len(lbls)):
            lblDic[lbls[i]] = i
            self._classes.append(lbls[i])
        b = np.zeros([len(y), len(lbls)])
        for i in range(len(y)):
            b[i, lblDic[y[i]]] = 1
        return b
        
                
    #Clean up resources
    def __del__(self):
        self.sess.close()
       
        
#Multi-Layer Perceptron for Regression
class MLPR:
    #Predicted outputs
    pred = None
    #The loss function
    loss = None
    #The optimization method
    optmzr = None
    #Max number of iterations
    mItr = None
    #Error tolerance
    tol = None
    #Tensorflow session
    sess = None
    #Input placeholder
    x = None
    #Output placeholder
    y = None
    #Boolean for toggling verbose output
    vrbse = None
    #Batch size
    batSz = None

    #The constructor
    #param layers: A list of layer sizes
    #param actvFn: The activation function to use: 'tanh', 'sig', or 'relu'
    #param learnRate: The learning rate parameter
    #param decay: The decay parameter
    #param maxItr: Maximum number of training iterations
    #param tol: Maximum error tolerated 
    #param batchSize: Size of training batches to use (use all if None)
    #param verbose: Print training information
    #param reg: Regularization weight
    def __init__(self, layers, actvFn = 'tanh', optmzr = 'adam', learnRate = 0.001, decay = 0.9, 
                 maxItr = 2000, tol = 1e-2, batchSize = None, verbose = False, reg = 0.001):
        #Parameters
        self.tol = tol
        self.mItr = maxItr
        self.vrbse = verbose
        self.batSz = batchSize
        #Input size
        self.x = tf.placeholder("float", [None, layers[0]])
        #Output size
        self.y = tf.placeholder("float", [None, layers[-1]])
        #Setup the weight and bias variables
        weight, bias = _CreateVars(layers)       
        #Create the tensorflow MLP model
        self.pred = _CreateMLP(self.x, weight, bias, _GetActvFn(actvFn))
        #Use L2 as the cost function
        self.loss = tf.reduce_sum(tf.nn.l2_loss(self.pred - self.y))
        #Use regularization to prevent over-fitting
        if(reg is not None):
            self.loss += _CreateL2Reg(weight, bias) * reg
        #Use ADAM method to minimize the loss function
        self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
        #Initialize all variables on the TF session
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        
    #Fit the MLP to the data
    #param A: numpy matrix where each row is a sample
    #param y: numpy matrix of target values
    def fit(self, A, y):
        m = len(A)
        #Begin training
        for i in range(self.mItr):
            #Batch mode or all at once
            if(self.batSz is None):
                self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
            else:
                for j in range(0, m, self.batSz):
                    batA, batY = _NextBatch(A, y, j, self.batSz)
                    self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
            err = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / m)
            if(self.vrbse):
                print("Iter {:5d}\t{:.8f}".format(i + 1, err))
            if(err < self.tol):
                break
    
    #Predict the output given the input (only run after calling fit)
    #param A: The input values for which to predict outputs 
    #return: The predicted output values (one row per input sample)
    def predict(self, A):
        if(self.sess == None):
            print("Error: MLP has not yet been fitted.")
            return None
        res = self.sess.run(self.pred, feed_dict={self.x:A})
        return res
        
    #Predicts the ouputs for input A and then computes the RMSE between
    #The predicted values and the actualy values
    #param A: The input values for which to predict outputs 
    #param y: The actual target values
    #return: The RMSE
    def score(self, A, y):
        scr = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / len(A))
        return scr
        
    #Clean-up resources
    def __del__(self):
        self.sess.close()
        
        
from StockPredictor import StockPredictor, ParseData, PlotData
from sklearn.neighbors import KNeighborsRegressor
#Used to get command line arguments
import sys
#Used to check validity of date
from datetime import datetime

#from TFMLP import MLPR
    
#Display usage information
def PrintUsage():
    print('Usage:\n')
    print('\tpython stocks.py <csv file> <start date> <end date> <D|W|M>')
    print('\tD: Daily prediction')
    print('\tD: Weekly prediction')
    print('\tD: Montly prediction')

#Main program
def Main(args):
    if(len(args) != 3 and len(args) != 4):
        PrintUsage()
        return
    #Test if file exists
    try:
        open(args[0])
    except Exception as e:
        print('Error opening file: ' + args[0])
        print(str(e))
        PrintUsage()
        return
    #Test validity of start date string
    try:
        datetime.strptime(args[1], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[1])
        PrintUsage()
        return
    #Test validity of end date string
    try:
        datetime.strptime(args[2], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[2])
        PrintUsage()
        return    
    #Test validity of final optional argument
    if(len(args) == 4):
        predPrd = args[3].upper()
        if(predPrd == 'D'):
            predPrd = 'daily'
        elif(predPrd == 'W'):
            predPrd = 'weekly'
        elif(predPrd == 'M'):
            predPrd = 'monthly'
        else:
            PrintUsage()
            return
    else:
        predPrd = 'daily'
    #Everything looks okay; proceed with program
    #Grab the data frame
    D = ParseData(args[0])
    #The number of previous days of data used
    #when making a prediction
    numPastDays = 16
    PlotData(D)
    #Number of neurons in the input layer
    i = numPastDays * 7 + 1
    #Number of neurons in the output layer
    o = D.shape[1] - 1
    #Number of neurons in the hidden layers
    h = int((i + o) / 2)
    #The list of layer sizes
    layers = [i, h, h, h, h, h, h, o]
    #R = MLPR(layers, maxItr = 1000, tol = 0.40, reg = 0.001, verbose = True)
    R = KNeighborsRegressor(n_neighbors = 5)
    sp = StockPredictor(R, nPastDays = numPastDays)
    #Learn the dataset and then display performance statistics
    sp.Learn(D)
    sp.TestPerformance()
    #Perform prediction for a specified date range
    P = sp.PredictDate(args[1], args[2], predPrd)
    #Keep track of number of predicted results for plot
    n = P.shape[0]
    #Append the predicted results to the actual results
    D = P.append(D)
    #Predicted results are the first n rows
    PlotData(D, range(n + 1))   
    return (P, n)
    

#Main entry point for the program
if __name__ == "__main__":
    #Main(sys.argv[1:])
    p, n = Main(['D:/Documents/Python Scripts/Stocks/yahoostock.csv', '2016-11-02', '2016-12-31', 'D'])