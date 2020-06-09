#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


state_e = [0.25,0.25,0.25,0.25]
state_5 = [0.05,0,0.95,0]
state_i = [0.4,0.1,0.1,0.4]


# In[3]:


input_sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"
input_state_path = "EEEEEEEEEEEEEEEEEE5IIIIIII$"


# In[4]:


def traverse(dna_element,state_path_element):
#     print(dna_element,state_path_element)
    retval = 0
    if(dna_element=='A'):
        if(state_path_element=="E"):
            retval = np.log(state_e[0])
        elif(state_path_element=="5"):
            retval = np.log(state_5[0])
        elif(state_path_element=="I"):
            retval = np.log(state_i[0])
        return retval
    elif(dna_element=='C'):
        if(state_path_element=="E"):
            retval = np.log(state_e[1])
        elif(state_path_element=="5"):
            retval = np.log(state_5[1])
        elif(state_path_element=="I"):
            retval = np.log(state_i[1])
        return retval
    elif(dna_element=='G'):
        if(state_path_element=="E"):
            retval = np.log(state_e[2])
        elif(state_path_element=="5"):
            retval = np.log(state_5[2])
        elif(state_path_element=="I"):
            retval = np.log(state_i[2])
        return retval
    elif(dna_element=='T'):
        if(state_path_element=="E"):
            retval = np.log(state_e[3])
        elif(state_path_element=="5"):
            retval = np.log(state_5[3])
        elif(state_path_element=="I"):
            retval = np.log(state_i[3])
        return retval
    else:
        return retval


# In[5]:


def state_path_probabilities(cur,nxt):
    if(cur=='E' and nxt=='E'):
        return np.log (0.9)
    if(cur=='E' and nxt=='5'):
        return np.log (0.1)
    if(cur=='5' and nxt=='I'):
        return np.log (1.0)
    if(cur=='I' and nxt=='I'):
        return np.log (0.9)
    if(cur=='I' and nxt=='$'):
        return np.log (0.1)
    return  0.0
    


# In[6]:


def compute(state_path):
#     print(len(state_path),state_path)
    log_probability = 0
    for i in range(len(state_path)):
        if(state_path[i]=='$'):
            break
        if(state_path[i]=='E'):
            log_probability += traverse(input_sequence[i],state_path[i])
        elif(state_path[i]=='5'):
            log_probability += traverse(input_sequence[i],state_path[i])
        elif(state_path[i]=='I'):
            log_probability += traverse(input_sequence[i],state_path[i])
#         print(i)
        
        log_probability += state_path_probabilities(state_path[i],state_path[i+1])
        if(state_path[i] == 'I' and state_path[i+1]=='$'):
            break
        
            
    if(np.isneginf(log_probability)):
        log_probability = -50
#     print("log_probability ",log_probability)
    return log_probability


# In[7]:


def generate_state_path():
    ret_list = []
    for i in range(1,26):
        state_path = ""
        for j in range(0,i):
            state_path += "E"
        state_path += "5"
        for j in range(i+1,26):
            state_path += "I"
        state_path += "$"
        ret_list.append(state_path)
#         print(type(state_path),state_path)
    return ret_list


# In[8]:


def plot_graph(all_logs):
    position = [i for i in range(1,26)]
    plt.xlabel('position of 5')
    plt.ylabel('log probability')
    plt.title('probability with different position $ , E , I')
    plt.plot(position, all_logs)
    plt.show()


# In[9]:


def start_main(compute_input):
    if(compute_input):
        ans = compute(input_state_path)
        print("log probability = ",ans)
    else:
        all_state_path = generate_state_path()
        all_logs = []
        for path in all_state_path:
            all_logs.append(compute(path))
        ans = float('-inf')
        print(all_logs)
        for log in all_logs:
            if(log> ans):
                ans = log
        print("maximum log probability = ",ans)
        plot_graph(all_logs)


# In[10]:


start_main(True)


# In[12]:


start_main(False)


# In[ ]:




