#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data 
from sklearn.preprocessing import MinMaxScaler
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import YearLocator, MonthLocator
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# # Q - 1 - 1 : Stock price prediction using RNN

# In[2]:


def feature_scaling(df):
    
#     df['open'] = MinMaxScaler().fit_transform(df.open.values.reshape(-1,1))
    df['volume'] = MinMaxScaler().fit_transform(df.volume.values.reshape(-1,1))
    df['average'] = MinMaxScaler().fit_transform(df.average.values.reshape(-1,1))
    return df


# In[3]:


def train_validation_split(df,time_steps = 10,split_percentage = 0.3):

     
    data = df.values
    new_data = []
    
    for index in range(data.shape[0] - time_steps): 
        new_data.append(data[index: index + time_steps])
    
    new_data = np.array(new_data)
    
    valid_set_size = int(split_percentage*data.shape[0]);  
    train_set_size = new_data.shape[0] - (valid_set_size);
    
    X_train = new_data[:train_set_size,:-1,:]
    Y_train = new_data[:train_set_size,-1,:]
    
    X_validation = new_data[train_set_size:train_set_size+valid_set_size,:-1,:]
    Y_validation = new_data[train_set_size:train_set_size+valid_set_size,-1,:]
    

    return X_train, Y_train, X_validation, Y_validation


# In[4]:


def data_preprocessing(isHMM = False):

    df = pd.read_csv("GoogleStocks.csv")
    df = df.sort_values(by=['date'])
    df["average"] = (df.low + df.high)/2
#     df.drop(['high','low','close','date'],axis=1,inplace=True)
    df.drop(['high','low','close','date','open'],axis=1,inplace=True)
    
    df = feature_scaling(df)
    
    if isHMM == True:

        train_set_percentage = 0.7
        train_set_size = int(df.shape[0]*train_set_percentage)

        train_df = df.iloc[:train_set_size,:].values
        validation_df = df.iloc[train_set_size:,:].values

        return train_df, validation_df
        
    else:
        return df


# In[5]:


def exploratory_data_visualization():
    df = data_preprocessing()
#     plt.plot(df.open.values, color='red', label='open')
    plt.plot(df.average.values, color='yellow', label='average')
    
    plt.title('stock price')
    plt.xlabel('time [days]')
    plt.ylabel('price')
    plt.legend(loc='best')
    plt.show()

    plt.plot(df.volume.values, color='blue', label='volume')
    plt.title('stock volume')
    plt.xlabel('time [days]')
    plt.ylabel('volume')
    plt.legend(loc='best');
    plt.show()


# In[6]:


def train_evaluate_model(params):

    X_train, Y_train, X_validation, Y_validation, n_steps,n_inputs,n_neurons,n_outputs,n_layers,learning_rate,n_epochs = params
    
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])

    # use Basic RNN Cell
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)for layer in range(n_layers)]

    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:,n_steps-1,:]

    loss = tf.reduce_mean(tf.square(outputs - y)) 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    training_op = optimizer.minimize(loss)

    # run graph
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            
            sess.run(training_op, feed_dict={X: X_train, y: Y_train}) 
            
            if epoch % 10 == 0:
                mse_train = loss.eval(feed_dict={X: X_train, y:Y_train}) 
                mse_valid = loss.eval(feed_dict={X: X_validation, y: Y_validation})
                
#                 print('%d epochs: MSE train/valid = %.8f/%.8f'%(epoch, mse_train, mse_valid))

        Y_train_prediction = sess.run(outputs, feed_dict={X: X_train})
        Y_valid_prediction = sess.run(outputs, feed_dict={X: X_validation})
        
    return  Y_train_prediction, Y_valid_prediction


# In[7]:


def output_visualization(Y_train,Y_validation,Y_train_prediction,Y_validation_prediction,time_steps,num_neurons,num_layers,index):
    
    plt.figure(figsize=(15, 5), dpi=100)
    
    x_length = Y_train.shape[0]
    
    plt.plot(np.arange(x_length), Y_train[:,1], color='blue', label='Train Target')

    plt.plot(np.arange(x_length, x_length + Y_validation.shape[0]), Y_validation[:,1],color='gray', label='Validation Target')

    x_length = Y_train_prediction.shape[0]
    
    plt.plot(np.arange(x_length),Y_train_prediction[:,1], color='red',label='Train Prediction')

    plt.plot(np.arange(x_length, x_length+Y_validation_prediction.shape[0]),Y_validation_prediction[:,1], color='orange', label='Valid Prediction')

     
    plt.title('Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price (normalized)')
    plt.legend(loc='best')
    plt.show()


# In[8]:


time_steps = 10
num_steps = time_steps-1 
num_inputs = 2
num_neurons = 200 
num_outputs = 2
num_layers = 2
learning_rate = 0.001
num_epochs = 300

hidden_layers = [2,3]
hidden_layer_size = [30,50,80]
time_steps_list = [20,50,75]

df = data_preprocessing()

X_train, Y_train, X_validation, Y_validation = train_validation_split(df,time_steps = time_steps,split_percentage = 0.2)

accuracy_train_list = []
accuracy_valid_list = []
number_of_layers = []
number_of_neurons = []
number_of_time_steps = []

index = 1

for num_layers in hidden_layers:
    
    for num_neurons in hidden_layer_size:
        
        for time_steps in time_steps_list:
    
            params = [X_train, Y_train, X_validation, Y_validation, num_steps,num_inputs,num_neurons,num_outputs,num_layers,learning_rate,num_epochs]

            Y_train_prediction,Y_validation_prediction = train_evaluate_model(params)

            print()

            output_visualization(Y_train,Y_validation,Y_train_prediction,Y_validation_prediction,time_steps,num_neurons,num_layers,index)
            index += 1
            
            accuracy_train = r2_score(Y_train[:,1],Y_train_prediction[:,1])
            accuracy_valid = r2_score(Y_validation[:,1],Y_validation_prediction[:,1])

            accuracy_train_list.append(accuracy_train)
            accuracy_valid_list.append(accuracy_valid)
            number_of_layers.append(num_layers)
            number_of_neurons.append(num_neurons)
            number_of_time_steps.append(time_steps)
            
            print("Model built with ",num_layers," layers and ",num_neurons," neurons at each layer and with time steps ",time_steps)
            print('R2 Score for open price for train data',accuracy_train)
            print('R2 Score for open price for validation data',accuracy_valid)
            


# # Q - 1 - 2 : Stock price prediction using Hidden Markov Model

# In[9]:


def exploratory_data_visualization():
    X, X_validation = data_preprocessing(isHMM = True)
    plt.figure(figsize=(15, 5), dpi=100) 
    plt.title("Stock open price for Google", fontsize = 14)
    plt.plot(np.arange(X.shape[0]),X[:,1])
    plt.ylabel("Price")
    plt.xlabel("Days in increasing order")
    plt.show()


# In[10]:


exploratory_data_visualization()


# In[11]:


def train_model(X,params):
    n_components, covariance_type, n_iter = params 
    model = GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
    model.fit(X)
    return model


# In[12]:


def model_visualization():
    
    X, X_validation = data_preprocessing(isHMM = True)
    # print(X)
    params = [15,"diag",1000]
    model = train_model(X,params)
    hidden_states = model.predict(X)

    # print(hidden_states)
    # print(model.monitor_)

    print("Has model converged: ",model.monitor_.converged)

    num_samples = X.shape[0]
    samples, _ = model.sample(num_samples)

    samples = MinMaxScaler().fit_transform(samples)
    open_norm = MinMaxScaler().fit_transform(X[:,1].reshape(-1,1))
    plt.figure(figsize=(20,10))
    plt.title("Stock open price for Google", fontsize = 14)
    plt.ylabel("Price (Normalized)")
    plt.xlabel("Days in increasing order")
    plt.plot(np.arange(num_samples),samples[:,1],color = 'green',label = "Generated by HMM")
    plt.plot(np.arange(num_samples),open_norm,color = 'blue',label = "Actual")
    plt.legend(loc='best')
    plt.show()


# In[13]:


model_visualization()


# In[16]:


def prediction(model,X,num_steps,return_prediction = False):
    
    hidden_states = model.predict(X)
    
    expected_values = np.dot(model.transmat_, model.means_)
    expected_values_columnwise = list(zip(*expected_values))
    expected_open = expected_values_columnwise[1]
    expected_volumes = expected_values_columnwise[0]
#     expected_average = expected_values_columnwise[2]
    
    predicted_open = []
    predicted_average = []
    predicted_volumes = []
    actual_volumes = []
    actual_open = []
    actual_average = []

    for i in range(num_steps):
        state = hidden_states[i]
        volume = X[i,0]#[0]
        open_val = X[i,1]#[1]
#         average = X[i,2]#[2]

        actual_volumes.append(volume)
        actual_open.append(open_val)
#         actual_average.append(average)

        predicted_open.append(expected_open[state])
        predicted_volumes.append(np.round(expected_volumes[state]))    
#         predicted_average.append(expected_average[state])

    if return_prediction == False:
        #Open
        plt.figure(figsize=(15, 5), dpi=100) 
        plt.title("Stock open price predition for Google", fontsize = 14)
        plt.plot(np.arange(num_steps),actual_open,label = "Actual")
        plt.plot(np.arange(num_steps),predicted_open, label = "Predicted")
        plt.ylabel("Price")
        plt.xlabel("Days in increasing order")
        plt.legend(loc = "best")
        plt.show()
    else: 
        return actual_open,predicted_open


# In[17]:


X, X_validation = data_preprocessing(isHMM = True)
n_steps_list = [20,50,75]
n_component = [4,8,12,30]
n_iterations = 1000
cov_type = "diag"
params = []

for i in n_steps_list:
    for j in n_component:
        print()
        print("Model based on number of hidden states: ",j," and time steps: ",i)
        print()
        params = [j,cov_type,n_iterations]
        model = train_model(X,params)
        print("On Train Data")
        prediction(model,X,i)
        print()
        print("On Validation Data")
        prediction(model,X_validation,i)
        print()


# ## Compare RNN and HMM Result

# In[18]:


def compare_results():
    
    X_HMM, X_HMM_validation = data_preprocessing(isHMM = True)
    time_steps = 200
    n_component = 50
    n_iterations = 1000
    cov_type = "tied"

    print("HMM Model based on number of hidden states: ",n_component," and time steps: ",time_steps)
    print()
    params = [n_component,cov_type,n_iterations]
    model = train_model(X_HMM,params)
    # print("On Train Data")
    Y_train_HMM,Y_train_prediction_HMM = prediction(model,X_HMM,time_steps,return_prediction = True)
    print()
    # print("On Validation Data")
    Y_validation_HMM,Y_validation_prediction_HMM = prediction(model,X_HMM_validation,time_steps,return_prediction = True)

    num_steps = time_steps-1 
    num_inputs = 2
    num_neurons = 100 
    num_outputs = 2
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 100

    hidden_layers = 3
    hidden_layer_size = 80

    df_RNN = data_preprocessing()
    X_train, Y_train, X_validation, Y_validation = train_validation_split(df,time_steps = time_steps,split_percentage = 0.2)

    params = [X_train, Y_train, X_validation, Y_validation, num_steps,num_inputs,num_neurons,num_outputs,num_layers,learning_rate,num_epochs]

    Y_train_prediction_RNN,Y_validation_prediction_RNN = train_evaluate_model(params)


    x_length = time_steps
    start = Y_train.shape[0] - time_steps
    
    
    Y_train_prediction_HMM = MinMaxScaler().fit_transform(np.array(Y_train_prediction_HMM).reshape(-1,1))
    Y_validation_prediction_HMM = MinMaxScaler().fit_transform(np.array(Y_validation_prediction_HMM).reshape(-1,1))
    
    plt.figure(figsize=(15, 5), dpi=100)
    plt.plot(np.arange(x_length), Y_train[start:,1], color='blue', label='Train Target')
    plt.plot(np.arange(x_length), Y_train_prediction_RNN[start:,1],color='green', label='Train Prediction using RNN')
    plt.plot(np.arange(x_length), Y_train_prediction_HMM,color='red', label='Train Prediction using HMM')

    plt.title('Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price (normalized)')
    plt.legend(loc='best')
    plt.show()
    plt.close()
    
    
#     print(Y_validation.shape)
#     print(Y_validation_prediction_HMM.shape)

    x_length = Y_validation.shape[0]    
    
    index = Y_validation_prediction_HMM.shape[0] - Y_validation.shape[0] 
    
    plt.figure(figsize=(15, 5), dpi=100)
    plt.plot(np.arange(x_length), Y_validation[:,1], color='blue', label='Validation Target')
    plt.plot(np.arange(x_length), Y_validation_prediction_RNN[:,1],color='green', label='Validation Prediction using RNN')
    plt.plot(np.arange(x_length), Y_validation_prediction_HMM[index:],color='red', label='Validation Prediction using HMM')

    plt.title('Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price (normalized)')
    plt.legend(loc='best')
    plt.show()
    plt.close()


# In[19]:


compare_results()

