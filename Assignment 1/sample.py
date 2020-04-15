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
import csv

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
month_to_data = {}  ## Dictionary (key:month , value:data)
for month in range(12):
    sample = np.empty(shape = (18 , 480))
    # 每個月18個空氣成分共有24*20筆data，故建立一18*480的array
    for day in range(20):
        for hour in range(24): 
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
            # 於sample放入第day天第hour小時的18筆空氣成分data
    month_to_data[month] = sample

# In[3]:
# Preprocess
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue  
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]

# In[4]:
# Normalization
mean = np.mean(x, axis = 0) 
std = np.std(x, axis = 0)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0 :
            x[i][j] = (x[i][j]- mean[j]) / std[j]
