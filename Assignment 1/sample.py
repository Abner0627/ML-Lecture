# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:38:39 2020

@author: Abner
"""

# In[1]:
# Import package
import sys
import numpy as np
import pandas as pd
import math
# import csv

# In[2]:
# Read in training set
raw_data = pd.read_csv('data/train.csv',encoding='big5')
# train.csv
raw_data = raw_data.to_numpy()
# 將dataframe轉為array
data = raw_data[:,3:]
# 取其第3個column之後的數值
data[data=="NR"] = 0
# 將NaN定為0
month_data = {}  ## Dictionary (key:month , value:data)
for month in range(12):
    sample = np.empty([18, 480])
    # 每個月18個空氣成分共有24*20筆data，故建立一18*480的array
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
        # 於sample放入第month月第day天的18筆空氣成分24小時的data
    month_data[month] = sample

# In[3]:
# Preprocess
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
# 宣告x為前9個小時18個空氣成分的data
# 每個月會有480個小時，每9小時形成一個data(1~9 2~10 3~11...471~480)，每個月會有471個data
y = np.empty(shape = (12 * 471 , 1),dtype = float)
# 宣告y為前9個小時PM2.5的data

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue
                # 跳過第20天第15小時(不含)以後的data(超出471)
            x[month * 471 + day * 24 + hour,:] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1)
            # 將第month月第day天第1~9、2~10、3~11...15~24小時的data放入x
            # reshape(1,-1):使每次取出的data(18*9)變為(1*162)    參見:https://blog.csdn.net/W_weiying/article/details/82112337
            y[month * 471 + day * 24 + hour,0] = month_data[month][9 ,day * 24 + hour + 9]
            # 將第month月第day天第10個小時的PM2.5data放入y

# In[4]:
# Normalization
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
x

# In[5]:
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

# In[6]:
# Training
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
w

# In[7]:
# Testing
testdata = pd.read_csv('data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x

# In[8]:
# Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y
