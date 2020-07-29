# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:52:56 2020
Data processing is same as sample code
@author: Lab
"""

#%% Import package
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
# import csv


#%% Read in training set
raw_data = pd.read_csv('../data/train.csv',encoding='big5')
# header    參見:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html?highlight=header
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


#%% Preprocess
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
            # 故1月第1天前1~9小時的data被放在x[0,0:163]的位置，且每9個column為一種空氣成分
            # reshape(1,-1):使每次取出的data(18*9)變為(1*162)    參見:https://numpy.org/doc/stable/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
            y[month * 471 + day * 24 + hour,0] = month_data[month][9 ,day * 24 + hour + 9]
            # 將第month月第day天第10個小時的PM2.5data放入y

#%% Normalization
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    # len(x)為x的row個數
    for j in range(len(x[0])): #18 * 9
        # len(x)為x的column個數
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

#%% Functions
# model
def model(x):
    return x.mm(w) + b

def lossfunc(ypre, ytar):
    MSE = nn.MSELoss()
    return torch.sqrt(MSE(ypre, ytar))

#%% Initialize parameters
dim = 18 * 9 + 1
w = torch.randn((dim, 1), requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 2.5
ep = 1000

#%% Optimizer
optimizer = optim.Adagrad([w, b], lr)

#%% Training
x_in = torch.from_numpy(x).float()
ytar = torch.from_numpy(y)

for t in range(ep):
    # Set the gradients to 0.
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    ypre = model(x_in).double()
    # See how far off the prediction is
    loss = lossfunc(ypre, ytar)
    # Compute the gradient of the loss with respect to A and b.
    loss.backward()   
    # Update A and b accordingly.
    optimizer.step()
    
    if (t%100 == 0 or t == ep - 1):
        print(str(t) + ":" + str(loss.detach().numpy()))

w_array = w.detach().numpy()
np.save('weight.npy', w_array)

#%% Testing
testdata = pd.read_csv('../data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
# 共240筆18個空氣成分前9個小時的data
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# Normalization
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)


#%% Prediction
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
np.save('ansy.npy', ans_y)
np.savetxt("ansy.csv", ans_y, delimiter=",")