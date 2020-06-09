#!/usr/bin/env python
# coding: utf-8

# In[1]:

import subprocess, sys
import numpy as np
import pandas as pd
import ssl
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import heapq
import itertools
import random
from pprint import pprint


# In[2]:


def data_preprocessing():
    
    global feature_list, feature_type
    
    feature_list = ["sepal length","sepal width", "petal length", "petal width","label"]
    
    df = pd.read_csv("Iris.csv", names = feature_list)

    train_df, test_df = train_test_split_data(df, 0.3)
    
    return train_df, test_df
    


# In[3]:


def train_test_split_data(df, size):
    
    if isinstance(size, float):
        size = round(size * len(df))
    
    #getting indexes of dataset in a list
    indices = df.index.tolist()
    
    #randomly choosing "size" number of indices for validation set
    indices = random.sample(population=indices, k=size)

    #Creating validation set
    validation_df = df.loc[indices]
    
    #Creating trianing set
    train_df = df.drop(indices)
    
    return train_df, validation_df


# In[4]:


def calculate_confusion_matrix(df):
    
    confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    
    unique_value = list(np.unique(df["label"].values))
    
    for index, row in df.iterrows():
        
        i = unique_value.index(row["label"])
        j = unique_value.index(row["result"])
        
        confusion_matrix[i][j] += 1
            
    confusion_matrix = np.asarray(confusion_matrix)
            
    return confusion_matrix
        


# In[5]:


def calculate_f1_score(df):
    
    precision = calculate_precision(df)
    recall = calculate_recall(df)
    
    #If recall and precision is both 0 then f1 score is undefined
    if precision == 0 or recall == 0:
        return 0
    
    #calculate f1 score
    f1_score = 2*((precision*recall)/(precision+recall))

    return f1_score


# In[6]:


def calculate_accuracy(df):
    
    #mean of all results
    accuracy = df["correct_result"].mean()
    
    return accuracy


# In[7]:


def precision(label, confusion_matrix):
    
    col = confusion_matrix[:, label]

    return confusion_matrix[label, label] / col.sum()


# In[8]:


def calculate_precision(df):
    
    confusion_matrix = calculate_confusion_matrix(df)
    
    rows, columns = confusion_matrix.shape
    
    sum_of_precisions = 0
    
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    
    return sum_of_precisions / rows


# In[9]:


def recall(label, confusion_matrix):
    
    row = confusion_matrix[label, :]

    return confusion_matrix[label, label] / row.sum()


# In[10]:


def calculate_recall(df):
    
    confusion_matrix = calculate_confusion_matrix(df)
    
    rows, columns = confusion_matrix.shape
  
    sum_of_recalls = 0
    
    for label in range(columns):
    
        sum_of_recalls += recall(label, confusion_matrix)
    
    return sum_of_recalls / columns


# In[11]:


def calculate_distance(test_row,train_row,distance_type):
    
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


# In[12]:


def classify_data(test_row, data, k,distance_type):

    distance_list = []
    
    for row in data:
    
        distance_list.append(calculate_distance(test_row,row,distance_type))
    
    
    k_smallest_point = heapq.nsmallest(k, distance_list)
    
    label_unique_values = []
    
    for item in k_smallest_point:
        
        label_unique_values.append(item[1])
        
        
    return max(label_unique_values,key=label_unique_values.count)
    


# In[13]:


def convert_string_to_float(test_row):
    
    if test_row[-1] == "Iris-setosa":
        return 0
    elif test_row[-1] == "Iris-virginica":
        return  1
    elif test_row[-1] == "Iris-versicolor":
        return  2


# In[21]:


def train_and_evaluate_model(test_filename):
    
    from sklearn import datasets 
    from sklearn.metrics import confusion_matrix 
    from sklearn.model_selection import train_test_split 
    from sklearn.neighbors import KNeighborsClassifier


    feature_list = ["sepal length","sepal width", "petal length", "petal width","label"]
    
    df = pd.read_csv("Iris.csv", names = feature_list)

    df["label"] = df.apply(convert_string_to_float, axis=1)
    
    X = df.values[:, :-1] 
    y = df.values[:, -1]

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
        custom_knn_accuracy.append(calculate_accuracy(validation_df))

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
    
    feature_list = ["sepal length","sepal width", "petal length", "petal width","label"]
    
    test_df = pd.read_csv(test_filename, names = feature_list)

    test_df["label"] = test_df.apply(convert_string_to_float, axis=1)

    X_test = test_df[test_df.columns[0:-1]]
    y_test = test_df[test_df.columns[-1]]
    
    custom_knn_accuracy_test = []
    sklearn_accuracy_test = []
    k_list_test = []
    
    for k in range(15):

        test_df["result"] = test_df.apply(classify_data, args=(train_data,k+1,"Hellinger"), axis=1)
        test_df["correct_result"] = test_df["result"] == test_df[test_df.columns[-2]]

        k_list_test.append(k+1)
        custom_knn_accuracy_test.append(calculate_accuracy(test_df))

        test_df.drop(["correct_result"],inplace=True,axis = 1)
        
        knn_model = KNeighborsClassifier(n_neighbors = k+1).fit(X_train, y_train)
        sklearn_accuracy_test.append(knn_model.score(X_test, y_test))

    accuracy_test = pd.DataFrame(
    {'K': k_list_test,
     'Custom KNN Model': sklearn_accuracy_test,
     'Scikit Learn Model': custom_knn_accuracy_test
    })

    #Accuracy visualisation
    
    accuracy_test = accuracy_test.melt('K', var_name='Implementation',value_name='Accuracy')
    accuracy_test_graph = sns.factorplot(x="K", y="Accuracy", hue='Implementation', data=accuracy_test)


# # Data Exploration and Visualization

# In[15]:


feature_list = ["sepal length","sepal width", "petal length", "petal width","Species"]
    
iris = pd.read_csv("Iris.csv", names = feature_list)
    
ax = iris[iris.Species=='Iris-setosa'].plot.scatter(x='sepal length', y='sepal width',color='red', label='setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='sepal length', y='sepal width',color='green', label='versicolor', ax=ax)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='sepal length', y='sepal width', color='blue', label='virginica', ax=ax)
ax.set_title("Iris Dataset")


# In[16]:


feature_list = ["sepal length","sepal width", "petal length", "petal width","Species"]
    
iris = pd.read_csv("Iris.csv", names = feature_list)
    
ax = iris[iris.Species=='Iris-setosa'].plot.scatter(x='petal length', y='petal width',color='red', label='setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='petal length', y='petal width',color='green', label='versicolor', ax=ax)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='petal length', y='petal width', color='blue', label='virginica', ax=ax)
ax.set_title("Iris Dataset")


# In[17]:


sns.pairplot(iris[['sepal length', 'sepal width', 'petal length', 'petal width', 'Species']],hue="Species", diag_kind="kde")


# In[18]:


def plot_metrics_vs_distance_type():
    
    k_list = []

    Euclidean_accuracy = []
    Euclidean_precision = []
    Euclidean_recall = []
    Euclidean_f1_score = []
    
    Manhattan_accuracy = []
    Manhattan_precision = []
    Manhattan_recall = []
    Manhattan_f1_score = []
    
    Chebyshev_accuracy = []
    Chebyshev_precision = []
    Chebyshev_recall = []
    Chebyshev_f1_score = []
    
    Hellinger_accuracy = []
    Hellinger_precision = []
    Hellinger_recall = []
    Hellinger_f1_score = []
    
    
    distance_type = ["Euclidean","Manhattan","Chebyshev","Hellinger"]
    
    train_df, validation_df = data_preprocessing()
    data = train_df.values

    
    for k in range(15):

        k_list.append(k+1)
        
        #Euclidean
        validation_df["result"] = validation_df.apply(classify_data, args=(data,k+1,"Euclidean"), axis=1)
        validation_df["correct_result"] = validation_df["result"] == validation_df["label"]
        
        Euclidean_accuracy.append(calculate_accuracy(validation_df))
        Euclidean_precision.append(calculate_precision(validation_df))
        Euclidean_recall.append(calculate_recall(validation_df))
        Euclidean_f1_score.append(calculate_f1_score(validation_df))
        
        #Manhattan
        validation_df["result"] = validation_df.apply(classify_data, args=(data,k+1,"Manhattan"), axis=1)
        validation_df["correct_result"] = validation_df["result"] == validation_df["label"]
        
        Manhattan_accuracy.append(calculate_accuracy(validation_df))
        Manhattan_precision.append(calculate_precision(validation_df))
        Manhattan_recall.append(calculate_recall(validation_df))
        Manhattan_f1_score.append(calculate_f1_score(validation_df))
        
        #Chebyshev
        validation_df["result"] = validation_df.apply(classify_data, args=(data,k+1,"Chebyshev"), axis=1)
        validation_df["correct_result"] = validation_df["result"] == validation_df["label"]
        
        Chebyshev_accuracy.append(calculate_accuracy(validation_df))
        Chebyshev_precision.append(calculate_precision(validation_df))
        Chebyshev_recall.append(calculate_recall(validation_df))
        Chebyshev_f1_score.append(calculate_f1_score(validation_df))
        
        #Hellinger
        validation_df["result"] = validation_df.apply(classify_data, args=(data,k+1,"Hellinger"), axis=1)
        validation_df["correct_result"] = validation_df["result"] == validation_df["label"]
        
        Hellinger_accuracy.append(calculate_accuracy(validation_df))
        Hellinger_precision.append(calculate_precision(validation_df))
        Hellinger_recall.append(calculate_recall(validation_df))
        Hellinger_f1_score.append(calculate_f1_score(validation_df))
    
    
    
    #Creating dataframe for accuracy data
    accuracy = pd.DataFrame(
    {'K': k_list,
     'Euclidean': Euclidean_accuracy,
     'Manhattan': Manhattan_accuracy,
     'Chebyshev': Chebyshev_accuracy,
     'Hellinger': Hellinger_accuracy
    })
    
    #Accuracy visualisation
    accuracy = accuracy.melt('K', var_name='Distance Measure',  value_name='Accuracy')
    accuracy_graph = sns.factorplot(x="K", y="Accuracy", hue='Distance Measure', data=accuracy)
    
    #Creating dataframe for precision data
    precision = pd.DataFrame(
    {'K': k_list,
     'Euclidean': Euclidean_precision,
     'Manhattan': Manhattan_precision,
     'Chebyshev': Chebyshev_precision,
     'Hellinger': Hellinger_precision
    })

    #Precision visualisation
    precision = precision.melt('K', var_name='Distance Measure',  value_name='Precision')
    precision_graph = sns.factorplot(x="K", y="Precision", hue='Distance Measure', data=precision)
    
    #Creating dataframe for recall data
    recall = pd.DataFrame(
    {'K': k_list,
     'Euclidean': Euclidean_recall,
     'Manhattan': Manhattan_recall,
     'Chebyshev': Chebyshev_recall,
     'Hellinger': Hellinger_recall
    })

    #Recall visualisation
    recall = recall.melt('K', var_name='Distance Measure',  value_name='Recall')
    recall_graph = sns.factorplot(x="K", y="Recall", hue='Distance Measure', data=recall)
    
    #Creating dataframe for f1 score data
    f1_score = pd.DataFrame(
    {'K': k_list,
     'Euclidean': Euclidean_f1_score,
     'Manhattan': Manhattan_f1_score,
     'Chebyshev': Chebyshev_f1_score,
     'Hellinger': Hellinger_f1_score
    })

    #F1 Score visualisation
    f1_score = f1_score.melt('K', var_name='Distance Measure',  value_name='F1 Score')
    f1_score_graph = sns.factorplot(x="K", y="F1 Score", hue='Distance Measure', data=f1_score)


# # Q 1 - 2 - 1

# In[22]:


test_filename = subprocess.list2cmdline(sys.argv[1:])
train_and_evaluate_model(test_filename)


# # Q 1 - 2 - 2

# In[20]:


plot_metrics_vs_distance_type()

