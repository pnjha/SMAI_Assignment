#!/usr/bin/env python
# coding: utf-8

# In[121]:

import subprocess, sys
import numpy as np
import pandas as pd
import ssl
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import itertools
import random
from pprint import pprint


# In[100]:


class catgorical_feature:
    def __init__(self,name):
        self.attribute_name = name
        self.children = {}


# In[101]:


class continuous_feature:
    def __init__(self,name):
        self.attribute_name = name
        self.mean = {}
        self.sd = {}


# In[102]:


def data_preprocessing():
    
    global feature_list, feature_type
    
    feature_list = ["ID","Age","Experience","Income","Zipcode","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Label","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]
    
    df = pd.read_csv("data.csv", names = feature_list,skiprows = 1)
    
    df["temp"] = df.Label
    df = df.drop(["Label"], axis=1)
    df["label"] = df.temp
    df = df.drop(["temp"], axis=1)
    
    df = df.drop(["ID","Zipcode"], axis=1)
    
    feature_list = ["Age","Experience","Income","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card","label"]
    
    feature_type = classify_features(df)
    
    pos_df = df.loc[df['label'] == 0]
    neg_df = df.loc[df['label'] == 1]
    
    #spliting positive and negative dataset into randomly 80-20 % split
    pos_train_df, pos_test_df = train_test_split_data(pos_df, 0.2)
    neg_train_df, neg_test_df = train_test_split_data(neg_df, 0.2)
    
    #merging positive and negative data split so that training and validation dataset contains equal number of positive and negative value of feature label 
    train_df = pd.concat([pos_train_df, neg_train_df])
    test_df = pd.concat([pos_test_df, neg_test_df])
    
    return train_df, test_df
    


# In[103]:


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


# In[104]:


def classify_features(df):
    
    feature_type = []
    
    unique_values_treshold = 5
    
    for feature in df.columns:
            
        #Find unique values of a features
        unique_values = df[feature].unique()
        value = unique_values[0]

        #if value is string then it has to be categorical or if no of unique values is less than threshold
        if (isinstance(value, str)) or (len(unique_values) <= unique_values_treshold):
            feature_type.append("categorical")
        else:
            feature_type.append("continuous")
    
    return feature_type
    


# In[105]:


def train_classifier(train_df):
    
    global feature_summary
    
    feature_summary = summarize_dataset(train_df.values)
    
    return feature_summary


# In[106]:


def summarize_dataset(data):
    
    feature_summary = []
    no_of_columns = data.shape[1]
    
    #Looping on all features and storing all unique values as split points except on left feature
    for column_index in range(no_of_columns-1):          
        
        if feature_type[column_index] == "categorical":
                
            categorical_obj = catgorical_feature(feature_list[column_index])

            values = data[:, column_index]
            unique_values = np.unique(values)

            for value in unique_values:

                categorical_obj.children[value] = calculate_probability_categorical(data,column_index,value)
            
            feature_summary.append(categorical_obj)      
                    
        elif feature_type[column_index] == "continuous":
        
            continuous_obj = continuous_feature(feature_list[column_index])
            
            values = data[:, -1]
            unique_values = np.unique(values)
            

            for value in unique_values:

                continuous_obj.mean[(value)] = (calculate_mean(data,column_index,value)) 
                continuous_obj.sd[(value)] = (calculate_sd(data,column_index,value))
            
            feature_summary.append(continuous_obj)
    
    return feature_summary
    


# In[107]:


def calculate_probability_categorical(data,column_index,feature_value):

    values_list = {}

    label_column = data[:, -1]
    
    label_unique_values , label_counts = np.unique(label_column, return_counts=True)


    for index in range(len(label_unique_values)):
        
        temp = data[label_column == label_unique_values[index]]
        
        column_values = temp[:, column_index]
        
        unique_values , counts = np.unique(column_values, return_counts=True)
        
        unique_values = list(unique_values)
        
        feature_index = unique_values.index(feature_value)
    
        count_feature_value = counts[feature_index]
    
        values_list[(label_unique_values[index])] = count_feature_value/label_counts[index]
    
    
    return values_list


# In[108]:


def calculate_probability_continuous(column_index,feature_value,label_value):

    mean = feature_summary[column_index].mean[(label_value)]
    sd = feature_summary[column_index].sd[(label_value)]
    
    exponent = math.exp(-(math.pow(feature_value-mean,2)/(2*math.pow(sd,2))))

    return (1/(math.sqrt(2*math.pi)*sd))*exponent


# In[109]:


def calculate_mean(data,column_index,label_value):
    
    label_column = data[:, -1]
    data = data[label_column == label_value]

    column_values = data[:, column_index]
    
    return np.mean(column_values,dtype=np.float64)
    


# In[110]:


def calculate_sd(data,column_index,label_value):
    
    label_column = data[:, -1]
    data = data[label_column == label_value]

    column_values = data[:, column_index]
    
    return np.std(column_values,dtype=np.float64)


# In[111]:


def classify_data(test_row,label_column):
    
    global feature_summary
    
    probablility_list = {}
    
    label_unique_values , counts = np.unique(label_column, return_counts=True)
    label_index = 0
    
    for label_unique in label_unique_values:
    
        probability = 1
    
        for index in range(len(feature_list)-1):
            
            if feature_type[index] == "categorical":
                
                probability *= feature_summary[index].children[(test_row[feature_list[index]])][(label_unique)]
                
            elif feature_type[index] == "continuous":

                probability *= calculate_probability_continuous(index,test_row[feature_list[index]],label_unique)
    
        probability *= counts[label_index]/counts.sum()
    
        probablility_list[(label_unique)] = probability
        
        label_index += 1
        
    sum_values = 0
    label_answer = ""
    max_value = float('-inf')
    
    for item, values in probablility_list.items():
        
        sum_values += values
        
    for item, values in probablility_list.items():
        if max_value < (float)(values/sum_values):
            
            max_value = (float)(values/sum_values)
            label_answer = (int(item))
            
    return label_answer


# In[112]:


def print_confusion_matrix(df):
    
    #to count true positive
    count_TP = 0
    
    #to count false positive
    count_FP = 0
    
    #to count false negative
    count_FN = 0
    
    #to count true negative
    count_TN = 0
    
    for index, row in df.iterrows():
        if row["result"] == row["label"] and row["label"] == 1:
            count_TP += 1
        elif row["result"] == row["label"] and row["label"] == 0:
            count_TN += 1    
        elif row["result"] == 1 and row["label"] == 0:
            count_FP += 1
        elif row["result"] == 0 and row["label"] == 1:    
            count_FN += 1
            
    print("True Positive: ", count_TP)
    print("True Negative: ", count_TN)
    print("False Positive: ", count_FP)
    print("False Negative: ", count_FN)


# In[113]:


def calculate_f1_score(df):
    
    precision = calculate_precision(df)
    recall = calculate_recall(df)
    
    #If recall and precision is both 0 then f1 score is undefined
    if precision == 0 or recall == 0:
        return 0
    
    #calculate f1 score
    f1_score = 2*((precision*recall)/(precision+recall))

    return f1_score


# In[114]:


def calculate_accuracy(df):
    
    #mean of all results
    accuracy = df["correct_result"].mean()
    
    return accuracy


# In[115]:


def calculate_precision(df):

    #to count true positive
    count_TP = 0
    
    #to count false positive
    count_FP = 0
    
    for index, row in df.iterrows():
        if row["result"] == row["label"] and row["label"] == 1:
            count_TP += 1
        elif row["result"] == 1 and row["label"] == 0:
            count_FP += 1
    
    #To check whether precision is defined or not. If not then return 0
    if count_TP == 0 and count_FP == 0 :
        return 0
    
    precision = (count_TP)/(count_TP + count_FP)
    
    return precision       


# In[116]:


def calculate_recall(df):
    
    #to count true positive
    count_TP = 0
    
    #to count false negative
    count_FN = 0
    
    for index, row in df.iterrows():
        if row["result"] == row["label"] and row["label"] == 1:
            count_TP += 1
        elif row["result"] == 0 and row["label"] == 1:    
            count_FN += 1
    
    #To check whether precision is defined or not. If not then return 0
    if count_TP == 0 and count_FN == 0 :
        return 0
    
    recall = (count_TP)/(count_TP + count_FN)
    
    return recall        


# In[117]:


def evaluate_model(filename):

    train_df,validation_df = data_preprocessing()

    data = train_df.values
    label_column = train_df.values[:,-1]
    feature_summary = train_classifier(train_df)

    validation_df["result"] = validation_df.apply(classify_data, args=(label_column,), axis=1)
    validation_df["correct_result"] = validation_df["result"] == validation_df["label"]

    print()
    print("Results on Validation Data")
    print()
    
    print("Confusion Matrix")
    print_confusion_matrix(validation_df)
    print("Accuracy: ",calculate_accuracy(validation_df))
    print("Precision: ",calculate_precision(validation_df))
    print("Recall: ",calculate_recall(validation_df))
    print("F1 Score: ",calculate_f1_score(validation_df))
    
    
    feature_list = ["ID","Age","Experience","Income","Zipcode","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Label","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]
    
    test_df = pd.read_csv(filename, names = feature_list)
    test_df["temp"] = test_df.Label
    test_df = test_df.drop(["Label"], axis=1)
    test_df["label"] = test_df.temp
    test_df = test_df.drop(["temp"], axis=1)
    
    test_df = test_df.drop(["ID","Zipcode"], axis=1)
    feature_list = ["Age","Experience","Income","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card","label"]

    
    
#     test_label_column = test_df.values[:,-1]
    
    test_df["result"] = test_df.apply(classify_data, args=(label_column,), axis=1)
    test_df["correct_result"] = test_df["result"] == test_df["label"]
    
    
    print()
    print("Results on Test Data")
    print()
    
    print("Confusion Matrix")
    print_confusion_matrix(test_df)
    print("Accuracy: ",calculate_accuracy(test_df))
    print("Precision: ",calculate_precision(test_df))
    print("Recall: ",calculate_recall(test_df))
    print("F1 Score: ",calculate_f1_score(test_df))


# # Using Scikit Learn Library 

# In[118]:


def train_and_evaluate_using_sklearn(filename):
    
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    
    feature_list = ["ID","Age","Experience","Income","Zipcode","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Label","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]

    df = pd.read_csv("data.csv", names = feature_list)

    df["temp"] = df.Label
    df = df.drop(["Label"], axis=1)
    df["label"] = df.temp
    df = df.drop(["temp"], axis=1)

    df = df.drop(["ID","Zipcode"], axis=1)

    feature_list = ["Age","Experience","Income","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]

    X_train, X_test = train_test_split(df, test_size=0.3, random_state=int(time.time()))

    gnb = GaussianNB()

    gnb.fit(
        X_train[feature_list].values,
        X_train["label"]
    )
    y_pred = gnb.predict(X_test[feature_list])

    print()
    print("Results on Validation Data")
    print()
    
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
          .format(
              X_test.shape[0],
              (X_test["label"] != y_pred).sum(),
              100*(1-(X_test["label"] != y_pred).sum()/X_test.shape[0])
    ))
    
    
    print()
    print("Results on Test Data")
    print()
    
    
    feature_list = ["ID","Age","Experience","Income","Zipcode","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Label","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]

    test_df = pd.read_csv("data.csv", names = feature_list)

    feature_list = ["Age","Experience","Income","Family_Size","Spending_Per_Month","Education_Level","Mortgage_Value","Securities_Account","Security_Deposits","Internet_Banking","Credit_Card"]

    
    test_df["temp"] = test_df.Label
    test_df = test_df.drop(["Label"], axis=1)
    test_df["label"] = test_df.temp
    test_df = test_df.drop(["temp"], axis=1)

    test_df = test_df.drop(["ID","Zipcode"], axis=1)
    
    gnb.fit(
        test_df[feature_list].values,
        test_df["label"]
    )
    
    y_pred_test = gnb.predict(test_df[feature_list])
    
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
          .format(
              test_df.shape[0],
              (test_df["label"] != y_pred_test).sum(),
              100*(1-(test_df["label"] != y_pred_test).sum()/test_df.shape[0])
    ))


# #  Q 1 - 2

# In[123]:


# test_file_name = sys.argv[1]

test_file_name = subprocess.list2cmdline(sys.argv[1:])

evaluate_model(test_file_name)

print()
print("Result using Sklearn Library")
print()

train_and_evaluate_using_sklearn(test_file_name)

