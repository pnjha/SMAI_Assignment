#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Implementation

# ## Import Packages

# In[8]:


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
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier


# ## Mean Normalization of Train Data

# In[9]:


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

# In[10]:


def scale_test_data(X_test,sd_mean_list):
    
    for test_row in X_test:
        
        for index in range(len(test_row)):

            mean = sd_mean_list[index][1]
            sd = sd_mean_list[index][0]

            test_row[index] = (test_row[index] - mean)/sd

    return X_test


# ## Data Preprocessing

# In[11]:


def data_preprocessing():
    
    
    df = pd.read_csv("admission_data.csv")
    
    df = df.drop(["Serial No."], axis = 1)
    
    length = df.values.shape[0]

    cols = df.columns.tolist()
    cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit ']
    df = df[cols]
    
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


# ## Draw Confusion Matrix

# In[12]:


def print_confusion_matrix(Y_predicted,Y_test):
    
    #to count true positive
    count_TP = 0
    
    #to count false positive
    count_FP = 0
    
    #to count false negative
    count_FN = 0
    
    #to count true negative
    count_TN = 0
    
    for predicted,actual in np.c_[Y_predicted,Y_test]:
        if predicted == actual and actual == 1:
            count_TP += 1
        elif predicted == actual and actual == 0:
            count_TN += 1    
        elif predicted == 1 and actual == 0:
            count_FP += 1
        elif predicted == 0 and actual == 1:    
            count_FN += 1
            
    print("True Positive: ", count_TP)
    print("True Negative: ", count_TN)
    print("False Positive: ", count_FP)
    print("False Negative: ", count_FN)


# ## Calculate F1 Score

# In[13]:


def calculate_f1_score(Y_predicted,Y_test):
    
    precision = calculate_precision(Y_predicted,Y_test)
    recall = calculate_recall(Y_predicted,Y_test)
    
    #If recall and precision is both 0 then f1 score is undefined
    if precision == 0 or recall == 0:
        return 0
    
    #calculate f1 score
    f1_score = 2*((precision*recall)/(precision+recall))

    return f1_score


# ## Calculate Accuracy

# In[14]:


def calculate_accuracy(Y_predicted,Y_test):
    accuracy = (Y_predicted==Y_test).mean()
    
    return accuracy


# ## Calculate Precision

# In[15]:


def calculate_precision(Y_predicted,Y_test):

    #to count true positive
    count_TP = 0
    
    #to count false positive
    count_FP = 0
    
    for predicted,actual in np.c_[Y_predicted,Y_test]:
        if predicted == actual and actual == 1:
            count_TP += 1    
        elif predicted == 1 and actual == 0:
            count_FP += 1
    
    #To check whether precision is defined or not. If not then return 0
    if count_TP == 0 and count_FP == 0 :
        return 0
    
    precision = (count_TP)/(count_TP + count_FP)
    
    return precision     


# ## Calculate Recall 

# In[16]:


def calculate_recall(Y_predicted,Y_test):
    
    #to count true positive
    count_TP = 0
    
    #to count false negative
    count_FN = 0
    
    for predicted,actual in np.c_[Y_predicted,Y_test]:
        if predicted == actual and actual == 1:
            count_TP += 1
        elif predicted == 0 and actual == 1:    
            count_FN += 1
    
    #To check whether precision is defined or not. If not then return 0
    if count_TP == 0 and count_FN == 0 :
        return 0
    
    recall = (count_TP)/(count_TP + count_FN)
    
    return recall 


# ## Calculate Specificity

# In[17]:


def calculate_specifity(Y_predicted,Y_test):
    #to count true positive
    count_TN = 0
    
    #to count false negative
    count_FN = 0
    
    for predicted,actual in np.c_[Y_predicted,Y_test]:
        if predicted == actual and actual == 0:
            count_TN += 1
        elif predicted == 0 and actual == 1:    
            count_FN += 1
    
    #To check whether precision is defined or not. If not then return 0
    if count_TN == 0 and count_FN == 0 :
        return 0
    
    specificity = (count_TN)/(count_TN + count_FN)
    
    return specificity


# ## Implementation of Logistic Regression Model

# In[18]:


class Logistic_regression_model:

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    
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
            
            y_predicted = self.calculate_sigmoid(y_inter)
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
                
    
    def predict(self, data, threshold = 0.5):
        data = self.concatenate_bias_column(data)
        return self.calculate_sigmoid(np.dot(data, self.parameters)) >= threshold


# In[19]:


def modify_label(data,threshold):

    data[data < threshold] = 0
    data[data >= threshold] = 1
    
    return data


# # Data Visualization

# In[20]:


def plot_error_iterations(iterations):

    sd_mean_list, x_train, x_test, y_train, y_test = data_preprocessing()

    threshold = 0.8

    y_train = modify_label(y_train, threshold)
    y_test = modify_label(y_test,threshold)
    
    model = Logistic_regression_model(0.1, iterations)
    model.fit(x_train, y_train)

    x_test = scale_test_data(x_test,sd_mean_list)

    y_predicted = model.predict(x_test,threshold)

    mean_value = mean(model.loss_value)
    sd = stdev(model.loss_value)

    # error = [(value)/mean_value for value in model.loss_value]

    error_vs_iterations = pd.DataFrame(
    {'Iterations': model.iteration,
     'Error': model.loss_value
    })

    error_vs_iterations = error_vs_iterations.melt('Iterations',  value_name='Error')
    error_vs_iterations_graph = sns.factorplot(x="Iterations", y="Error", data = error_vs_iterations)

    error_vs_iterations_graph.savefig("error_vs_iterations_graph.png")


# In[21]:


def plot_threshold_precision_recall(increment):

    sd_mean_list, X_train, X_test, Y_train, Y_test = data_preprocessing()
    

    threshold_current = 0

    threshold = []
    precision = []
    recall = []
    accuracy = []
    f1_score = []
    specificity = []


    while threshold_current < 0.9:

        #print("Iteration",threshold_current)
        threshold_current = threshold_current + increment
        threshold.append(round(threshold_current,1))

        x_train = copy.deepcopy(X_train)
        y_train = copy.deepcopy(Y_train)
        x_test = copy.deepcopy(X_test)
        y_test = copy.deepcopy(Y_test)

        # print(y_train)

        y_train = modify_label(y_train, threshold_current)
        y_test = modify_label(y_test,threshold_current)        

        # print(Y_train)

        model = Logistic_regression_model(0.1,10000)
        model.fit(x_train, y_train)

        x_test = scale_test_data(x_test,sd_mean_list)

        y_predicted = model.predict(x_test,threshold_current)

        # print_confusion_matrix(y_predicted,y_test)

        f1_score.append(calculate_f1_score(y_predicted,y_test))
        accuracy.append(calculate_accuracy(y_predicted,y_test))
        precision.append(calculate_precision(y_predicted,y_test))
        recall.append(calculate_recall(y_predicted,y_test))
        specificity.append(calculate_specifity(y_predicted,y_test))

        del y_train
        del y_test
        del x_train
        del x_test

    
    threshold_df = pd.DataFrame(
    {'Threshold': threshold,
     'Accuracy': accuracy,
     'Precision': precision,
     'Recall': recall,
     'Specificity':specificity,
     'F1 Score': f1_score     
    })
    
    #Accuracy visualisation
    threshold_df = threshold_df.melt('Threshold', var_name='Metrics',  value_name='Variations')
    threshold_graph = sns.factorplot(x="Threshold", y="Variations", hue='Metrics', data=threshold_df)

    threshold_graph.savefig("threshold_graph.png")





# ## Train and Evaluate Logistic Model

# In[22]:


def train_evaluate_logistic_model():
    
    increment = 0.1
    plot_threshold_precision_recall(increment)
    plot_error_iterations(10000)
    sd_mean_list, X_train, X_test, Y_train, Y_test = data_preprocessing()

    threshold = 0.8

    Y_train = modify_label(Y_train,threshold)
    Y_test = modify_label(Y_test,threshold)


    model = Logistic_regression_model(0.1,10000)
    model.fit(X_train, Y_train)

    X_test = scale_test_data(X_test,sd_mean_list)

    yPrediction = model.predict(X_test,threshold)

    print("Confusion Matrix")
    print()
    print_confusion_matrix(yPrediction,Y_test)
    print()
    print("Accuracy: ",(yPrediction==Y_test).mean())
    print("Precision: ",calculate_precision(yPrediction,Y_test))
    print("Recall: ",calculate_recall(yPrediction,Y_test))
    print("Specifity: ",calculate_specifity(yPrediction,Y_test))
    print("F1 Score: ",calculate_f1_score(yPrediction,Y_test))


# ## KNN Model

# ## Distance Measures

# In[23]:


def calculate_distance(test_row,train_row, distance_type):
    
    distance = 0
    label = train_row[-1]
    
    if distance_type == "Euclidean":
        for index in range(len(train_row)-1):

            distance += math.pow((float(test_row[index])) - (float(train_row[index])), 2)

        distance = math.sqrt(distance)

        distance_label_tuple = (distance, label)
    
    elif distance_type == "Manhattan":
        
        for index in range(len(train_row)-1):

            distance += abs((float(test_row[index])) - (float(train_row[index])))

        distance_label_tuple = (distance, label)
        
    elif distance_type == "Chebyshev":
        
        max_dist = float('-inf')
        
        for index in range(len(train_row)-1):
            
            distance = abs((float(test_row[index])) - (float(train_row[index])))
            
            if distance > max_dist:
                max_dist = distance
                
        distance_label_tuple = (max_dist, label)
        
    elif distance_type == "Hellinger":
        
        for index in range(len(train_row)-1):

            distance += math.pow(math.sqrt(float(test_row[index])) - math.sqrt(float(train_row[index])), 2)

        distance = math.sqrt(distance)*(1/math.sqrt(2))

        distance_label_tuple = (distance, label)
    
    return distance_label_tuple


# ## Classify test data based on training example

# In[24]:


def classify_data(test_row, data, k ,distance_type):

    distance_list = []
    
    for row in data:
    
        distance_list.append(calculate_distance(test_row,row,distance_type))
    
    
    k_smallest_point = heapq.nsmallest(k, distance_list)
    
    label_unique_values = []
    
    for item in k_smallest_point:
        
        label_unique_values.append(item[1])
        
    return max(label_unique_values,key=label_unique_values.count)
    


# ## Calculate Accuracy

# In[25]:


def calculate_accuracy_knn(df):
    
    #mean of all results
    accuracy = df["correct_result"].mean()
    
    return accuracy


# ## Train and Evaluate KNN Model

# In[26]:


def train_evaluate_model(test_filename,threshold):

    feature_list = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'label']

    df = pd.read_csv("admission_data.csv", names = feature_list,skiprows=1)

    X = df.values[:, :-1] 
    y = df.values[:, -1]

    y = modify_label(y,threshold)

    
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, random_state = 0) 

    train_data = X_train
    train_data[:,-1] = y_train

    validation_data = X_validation
    validation_data[:,-1] = y_validation

    validation_df = pd.DataFrame(validation_data)
    
    custom_knn_accuracy = []
    sklearn_accuracy = []
    k_list = []

    for k in range(15):

        validation_df["result"] = validation_df.apply(classify_data, args=(train_data,k+1,"Hellinger"), axis=1)
        validation_df["correct_result"] = validation_df["result"] == validation_df[validation_df.columns[-2]]

        k_list.append(k+1)
        custom_knn_accuracy.append(calculate_accuracy_knn(validation_df))

        validation_df.drop(["correct_result"],inplace=True,axis = 1)

        knn = KNeighborsClassifier(n_neighbors = k+1).fit(X_train, y_train)
        sklearn_accuracy.append(knn.score(X_validation, y_validation))

    accuracy = pd.DataFrame(
    {'K': k_list,
     'Custom KNN Model': custom_knn_accuracy,
     'Scikit Learn Model': sklearn_accuracy
    })

    print("Plot on Validation data")
    #Accuracy visualisation
    accuracy = accuracy.melt('K', var_name='Implementation',value_name='Accuracy')
    accuracy_graph = sns.factorplot(x="K", y="Accuracy", hue='Implementation', data=accuracy)

    
    
    
    print("Plot on Test data")
    
    feature_list = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'label']

    test_df = pd.read_csv(test_filename, names = feature_list,skiprows=1)
    
    X_test = test_df[test_df.columns[0:-1]].values
    y_test = test_df[test_df.columns[-1]].values
    
    y_test = modify_label(y_test,threshold)
    
    custom_knn_accuracy_test = []
    sklearn_accuracy_test = []
    k_list_test = []
    
    for k in range(15):

        test_df["result"] = test_df.apply(classify_data, args=(train_data,k+1,"Hellinger"), axis=1)
        test_df["correct_result"] = test_df["result"] == test_df[test_df.columns[-2]]

        k_list_test.append(k+1)
        custom_knn_accuracy_test.append(calculate_accuracy_knn(test_df))

        test_df.drop(["correct_result"],inplace=True,axis = 1)

        knn_model = KNeighborsClassifier(n_neighbors = k+1).fit(X_train, y_train)
        sklearn_accuracy_test.append(knn_model.score(X_test, y_test))

    accuracy_test = pd.DataFrame(
    {'K': k_list_test,
     'Custom KNN Model': custom_knn_accuracy_test,
     'Scikit Learn Model': sklearn_accuracy_test
    })

    #Accuracy visualisation
    
    accuracy_test = accuracy_test.melt('K', var_name='Implementation',value_name='Accuracy')
    accuracy_test_graph = sns.factorplot(x="K", y="Accuracy", hue='Implementation', data=accuracy_test)
    
    


# ## Q - 2 - 1 and Q - 2 - 3

# In[28]:


train_evaluate_logistic_model()


# Choice of threshold value depends largely on the use case of the machine learning model i.e. for some use cases we try to maximize true positive rate or recall or sensitivity and in some cases we try to maximize true negative rate or specificity. More so we might even try to maximize both recall and specificity. For the later case we can simply plot values of recall and specificity based on different threshold value and as we can observe from the above graph the point they meet maximize both so we take threshold where these two curves meet.

# ## Q - 2 - 2

# In[27]:


filename = "admission_data.csv"
threshold = 0.75
train_evaluate_model(filename,threshold)

