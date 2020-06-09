#!/usr/bin/env python
# coding: utf-8

# In[17]:


import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import random


# In[110]:


def generate__weight_matrix(size, scale = 1.0):

    stddev = scale/np.sqrt(np.prod(size))
    
    return np.random.normal(loc = 0, scale = stddev, size = size)


# In[79]:


def initializeWeight(size):

    return np.random.rand(size) * 0.01


# In[4]:


def max_pooling(img ,f = 2 ,s = 2):
   
    channels, height_img, weight_img = img.shape
    
    h = int((height_img - f)/s)+1 
    w = int((weight_img - f)/s)+1
    
    max_pool_output = np.zeros((channels, h, w)) 
    
    for i in range(channels):
        y = mpo_y = 0
    
        while y + f <= height_img:
            x = out_x = 0
    
            while x + f <= weight_img:
    
                max_pool_output[i, mpo_y, out_x] = np.max(img[i, y:y+f, x:x+f])
                x += s
                out_x += 1
    
            y += s
            mpo_y += 1
    
    return max_pool_output


# In[101]:


def convolution(img, filters, bias, stride = 1,toprint = 0):
    
    if toprint == 1:
        print(img.shape)
        print(filt.shape)
    
    no_of_filters, filter_height, filter_width,filter_channel = filters.shape 
    img_height, img_width,img_channel = img.shape 
    
    output_dim = int((img_height - filter_height)/stride)+1
    
    conv_out = np.zeros((no_of_filters,output_dim,output_dim))
    

    for filt in range(no_of_filters):
        if toprint == 1:
            print("#######################################")
            print(filt)
        y = mpo_y = 0

        while y + filter_height <= img_height:
            x = out_x = 0
 
            while x + filter_height <= img_height:

                if toprint == 1:
                    print(filters[filt].shape)
                    print(img[y:y+filter_height, x:x+filter_height,:].shape)
                    print(y,y+filter_height, x,x+filter_height)
                    
                conv_out[filt, mpo_y, out_x] = (np.sum(filters[filt] * img[y:y+filter_height, x:x+filter_height,:]) + bias[filt])/(filter_height**2)
                x += stride
                out_x += 1
            
            y += stride
            mpo_y += 1
        
    return conv_out


# In[6]:


def relu(cov_img):
    
    img_list = []
    
    for img in cov_img:
        
        img_list.append(np.maximum(img, 0))
    
    
    img_list = np.asarray(img_list)
    
#     print(img_list.shape)
    
#     for img in img_list:
        
#         ski.io.imshow(img)
#         ski.io.show()
        
    return img_list


# In[7]:


def sigmoid(cov_img):
    
    img_list = []
    
    for img in cov_img:
        
        img_list.append(np.nan_to_num(1.0 / (1.0 + np.exp(-img))))
    
    
    img_list = np.asarray(img_list)
        
    return img_list


# In[8]:


def tanh(cov_img):
    
    img_list = []
    
    for img in cov_img:
        
        img_list.append(np.nan_to_num(2.0/(1.0 + np.exp(-(2*img))) - 1))
    
    
    img_list = np.asarray(img_list)
        
    return img_list


# In[118]:


def draw_img(img_list):

    for img in img_list:
        
        plt.imshow(img)
        plt.show()


# In[143]:


def activation_function(arr,fuct):
    
    if fuct == "relu":
        
        return np.maximum(arr, 0)
    
    elif fuct == "sigmoid":
        
        return np.nan_to_num(1.0 / (1.0 + np.exp(-arr)))
    
    elif fuct == "tanh":
        
        return np.nan_to_num(2.0/(1.0 + np.exp(-(2*arr))) - 1)
    
    elif fuct == "softmax":
        
        e_x = np.exp(arr - np.max(arr))
        
        return e_x / e_x.sum(axis=0)


# In[145]:


def train(img):
    
    
    
    # A 5x5 filter will reduce the 32x32 image into 28x28 image
    l1_f = generate__weight_matrix([6, 5, 5, 3])

    # A 5x5 filter will reduce the 14x14 image into 10x10 image
    l2_f = generate__weight_matrix([16, 5, 5, 6])

    b1_c = generate__weight_matrix([6])
    b2_c = generate__weight_matrix([16])
        
#     print(l1_f)
#     print(l2_f)
    
    conv1 = convolution(img,l1_f,b1_c)
    print(img.shape)
    print(conv1.shape)
    
#     draw_img(conv1)
    print("############################################################################")
    
    
    relu1 = relu(conv1)
#     draw_img(relu1)
    print(relu1.shape)
    print("############################################################################")
    
#     relu1 = sigmoid(conv1)
#     draw_img(relu1)
    
#     relu1 = tanh(conv1)
#     draw_img(relu1)

    max_pool1 = max_pooling(relu1).T
    print(max_pool1.shape)
#     draw_img(max_pool1.T)
    print("############################################################################")
    
    conv2 = convolution(max_pool1,l2_f,b2_c)
    print(conv2.shape)
#     draw_img(conv2)
    print("############################################################################")
    
    relu2 = relu(conv2)
    print(relu2.shape)
#     draw_img(relu2)
    print("############################################################################")

    max_pool2 = max_pooling(relu2).T
    print(max_pool2.shape)
#     draw_img(max_pool2.T)
    print("############################################################################")

    nn_input = max_pool2.flatten()
    print(nn_input.shape)

    mean, sd = np.mean(nn_input), np.std(nn_input)
    nn_input = (nn_input-mean)/sd

    
    outputNodes = 10
    
    
    W1 = generate__weight_matrix([nn_input.shape[0],120])
    W2 = generate__weight_matrix([120,84])
    W3 = generate__weight_matrix([84,10])
    
    b1 = generate__weight_matrix([120])
    b2 = generate__weight_matrix([84])
    b3 = generate__weight_matrix([10])
    
    A_H1 = activation_function((np.dot(W1.T,nn_input) + b1),"sigmoid")
    A_H2 = activation_function((np.dot(W2.T,A_H1) + b2),"sigmoid")
    A_OUT = activation_function((np.dot(W3.T,A_H2) + b3),"softmax")
    print(np.asmatrix(A_OUT).T)
    print("Prediction: ",np.argmax(A_OUT))


# In[148]:


img = ski.io.imread("2.jpeg")
train(img)

