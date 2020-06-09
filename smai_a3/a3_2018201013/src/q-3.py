#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression One vs One and One vs All Model Implementation

# ## Import Packages 

# In[27]:


import subprocess, sys
import numpy as np
import pandas as pd
import ssl
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import heapq
import itertools
import random
import copy
from statistics import mean , stdev
from pprint import pprint
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


# ## Mean Normalization of Train Data

# In[28]:


def feature_scaling(train_data):
    
    no_of_columns = train_data.shape[1]
    
    sd_mean_list = []
    
    for index in range(no_of_columns):

        sd_val = np.std(train_data[:,index])
        mean_val = np.mean(train_data[:,index])
        train_data[:,index] = (train_data[:,index] - mean_val)/(sd_val)
        
        sd_mean_list.append([sd_val,mean_val])
        
    return sd_mean_list, train_data


# ## Mean Normalization of Test Data

# In[29]:


def scale_test_data(X_test,sd_mean_list):
    
    for test_row in X_test:
        
        for index in range(len(test_row)):

            mean = sd_mean_list[index][1]
            sd = sd_mean_list[index][0]

            test_row[index] = (test_row[index] - mean)/sd

    return X_test


# ## Data Preprocessing

# In[30]:


def data_preprocessing():
    
    feature_list = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    
    df = pd.read_csv("wine_datset.csv",names = feature_list, dtype=np.float64, skiprows=1,sep=";")

    cols = df.columns.tolist()

    X = df.iloc[:,0:-1]
    Y = df.iloc[:,-1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    Y_train = pd.DataFrame(Y_train)
    Y_train = Y_train.values
    
    Y_test = pd.DataFrame(Y_test)
    Y_test = Y_test.values
    
    sd_mean_list, X_train = feature_scaling(X_train.values)
    
    X_test = X_test.values


    return sd_mean_list, X_train, X_test, Y_train, Y_test 


# ## Calculate Accuracy

# In[31]:


def calculate_accuracy(Y_predicted,Y_test):
    
    count = 0
    total = 0
    for i in range(len(Y_predicted)):
        if Y_test[i] == Y_predicted[i]:
            count += 1
    
    accuracy = count/len(Y_predicted)
#     accuracy = (Y_predicted==Y_test).mean()
    
    return accuracy


# ## Logistic Regression Model Implementation

# In[32]:


class Logistic_regression_model:

    def __init__(self, learning_rate, iterations,threshold = 0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
    
    def calculate_sigmoid(self, exponent):
        return 1/(1 + np.exp(-exponent))

    def logistic_loss_function(self,y_predicted, y_actual):
        return (-y_actual*np.log(y_predicted) - (1-y_actual) * np.log(1 - y_predicted)).mean()
    

    def concatenate_bias_column(self, data):
        constants = np.ones((data.shape[0], 1))
        return np.concatenate((constants, data), axis=1)



    def fit(self, train_data, y_actual):
        
        self.loss_value = []
        self.iteration = []

        train_data = self.concatenate_bias_column(train_data)
        
        self.parameters = np.zeros(train_data.shape[1])
        
        
        for index in range(self.iterations):
            
            y_inter = np.dot(train_data, self.parameters)
        
            y_predicted = self.calculate_sigmoid(y_inter)# >= self.threshold
            
            y_predicted = pd.DataFrame(y_predicted)
            y_predicted = y_predicted.values

            gradient = np.dot(train_data.T, (y_predicted - y_actual)) / y_actual.size
            
            self.parameters = pd.DataFrame(self.parameters)
            self.parameters = self.parameters.values

            self.parameters = self.parameters - self.learning_rate * gradient
            

            if(index%10 == 0 and index<150):
                
                y_inter = np.dot(train_data, self.parameters)
                y_predicted = self.calculate_sigmoid(y_inter)
                self.loss_value.append(self.logistic_loss_function(y_predicted, y_actual))
                self.iteration.append(index)
                # print(self.logistic_loss_function(y_predicted, y_actual))
                
    
    def predict(self, data_row, method):

        data = data_row.tolist()
        data = [1] + data

        data = np.asmatrix(data)

        if method == "ova":
            return self.calculate_sigmoid(np.dot(data, self.parameters))
        elif method == "ovo":
            return self.calculate_sigmoid(np.dot(data, self.parameters)) >= self.threshold


# ## Logistic Regression One vs All Implementation 

# In[33]:


class logistic_regression_one_vs_all:
    
    def __init__(self, learning_rate, iterations,threshold):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
    
    def fit(self,X_train,Y_train):

        unique_classes, counts = np.unique(Y_train,return_counts=True)

        self.class_model_dict = {}

        for a_class in unique_classes:

            Y_train_modified = np.where(Y_train == a_class,1,0)
        
            self.class_model_dict[a_class] = Logistic_regression_model(self.learning_rate,self.iterations,self.threshold)
            self.class_model_dict[a_class].fit(X_train, Y_train_modified)

    def predict(self,X_test):
        
        y_predicted = []

        for data_row in X_test:

            max_probabilty = float('-inf')
            class_label = -1

            for a_class,model in self.class_model_dict.items():

                label_probability = model.predict(data_row,"ova")

                if label_probability > max_probabilty:

                    max_probabilty = label_probability
                    class_label = a_class

            y_predicted.append(class_label)

        y_predicted = np.asarray(y_predicted)
    
        return y_predicted
    


# ## Train and Evaluate Logistic Model One vs All

# In[34]:


def train_evaluate_logistic_one_vs_all():
    
    sd_mean_list, X_train, X_test, Y_train, Y_test = data_preprocessing()
    X_test = scale_test_data(X_test,sd_mean_list)

    lr_model = logistic_regression_one_vs_all(0.1,5000,0.7)

    lr_model.fit(X_train,Y_train)

    y_predicted = lr_model.predict(X_test)

    confusion_matrix = metrics.cluster.contingency_matrix(Y_test, y_predicted)

    print("One vs All Logistic Regression Model using Custom Implementation")
    print("Confusion Matrix")
    print(confusion_matrix)
    print()
    print("Accuracy: ",calculate_accuracy(y_predicted,Y_test))
    print()
    
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,Y_train)
    y_pre = clf.predict(X_test)

    confusion_matrix = metrics.cluster.contingency_matrix(Y_test,y_pre)
    print("One vs All Logistic Regression Model using Sklearn Multinomial Model")
    print("Confusion Matrix")
    print(confusion_matrix)
    print()
    print("Accuracy: ",calculate_accuracy(y_pre,Y_test))
    print()
#     clf.score(X_test, Y_test)


# ## Logistic Regression One vs One Implementation 

# In[35]:


class logistic_regression_one_vs_one:
    
    def __init__(self,learning_rate,iterations,threshold):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
        self.class_model_dict = {}

    def fit(self,X_train,Y_train):

        unique_classes, counts = np.unique(Y_train,return_counts=True)
        
        i = 0
        
        for a_class in unique_classes:

            j = i+1
             
            for b_class in unique_classes:

                if j > i:
                    
                    indices = [i for i in range(len(Y_train)) if Y_train[i] != a_class and Y_train[i] != b_class]

                    Y_train_modified = np.delete(Y_train, indices, 0)
                    
                    Y_train_modified = np.where(Y_train_modified == a_class,1,0)

                    X_train_modified = np.delete(X_train,indices,0)
                    
                    self.class_model_dict[(a_class,b_class)] = Logistic_regression_model(self.learning_rate,self.iterations,self.threshold)
                    
                    self.class_model_dict[(a_class,b_class)].fit(X_train_modified, Y_train_modified)
                    
            
                j += 1
            
            i += 1
        
    def predict(self,X_test):
    
        y_predicted = []
        
        for row in X_test:
            
            label_votes = []
        
            for label_tuple, model in self.class_model_dict.items():
            
                label_prediction = model.predict(row,"ovo")
        
                if label_prediction == 1:
                
                    label_votes.append(label_tuple[0])
                else:
                    label_votes.append(label_tuple[1])
                    
            y_predicted.append(max(set(label_votes), key=label_votes.count))
            
        y_predicted = np.asarray(y_predicted)
    
        return y_predicted


# ## Train and Evaluate Logistic Model One vs One

# In[36]:


def train_evaluate_logistic_one_vs_one():
    
    sd_mean_list, X_train, X_test, Y_train, Y_test = data_preprocessing()
    X_test = scale_test_data(X_test,sd_mean_list)

    lrovo_model = logistic_regression_one_vs_one(0.001,5000,0.7)

    lrovo_model.fit(X_train,Y_train)

    y_predicted = lrovo_model.predict(X_test)

    confusion_matrix = metrics.cluster.contingency_matrix(Y_test, y_predicted)

    print("One vs One Logistic Regression Model using Custom Implementation")
    print("Confusion Matrix")
    print()
    print(confusion_matrix)
    print()
    print("Accuracy: ",calculate_accuracy(y_predicted,Y_test))
    
    clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train,Y_train)
    y_pre = clf.predict(X_test)

    confusion_matrix = metrics.cluster.contingency_matrix(Y_test,y_pre)
    print("One vs One Logistic Regression Model using Sklearn Multinomial Model")
    print("Confusion Matrix")
    print(confusion_matrix)
    print()
    print("Accuracy: ",calculate_accuracy(y_pre,Y_test))
    print()


# ## Q - 3

# In[37]:


train_evaluate_logistic_one_vs_all()
train_evaluate_logistic_one_vs_one()

