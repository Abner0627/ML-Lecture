# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:29:39 2020
@author: Abner
"""


#%% Preparing Data
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# =============================================================================
# X_train_fpath = './data/X_train'
# Y_train_fpath = './data/Y_train'
# X_test_fpath = './data/X_test'
# output_fpath = './output_{}.csv'
# 
# # Parse csv files to numpy array
# with open(X_train_fpath) as f:
#     next(f)
#     X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
# with open(Y_train_fpath) as f:
#     next(f)
#     Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
# with open(X_test_fpath) as f:
#     next(f)
#     X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
# =============================================================================

# =============================================================================
# np.save('X_train', X_train)
# np.save('Y_train', Y_train)
# np.save('X_test', X_test)
# =============================================================================
output_fpath = './output_{}.csv'
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')


#%% Functions
def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    '''
     This function normalizes specific columns of X.
     The mean and standard variance of training data will be reused when processing testing data.
    
     Arguments:
         X: data to be processed
         train: 'True' when processing training data, 'False' for testing data
         specific_column: indexes of the columns that will be normalized. If 'None', all columns
             will be normalized.
         X_mean: mean value of training data, used when train = 'False'
         X_std: standard deviation of training data, used when train = 'False'
     Outputs:
         X: normalized data
         X_mean: computed mean value of training data
         X_std: computed standard deviation of training data
    '''

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
        # 假設X有N維column，則建立0~N-1的array
        # 參見:https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=arange#numpy.arange
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column] ,0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)    # 避免為0
    return X, X_mean, X_std

def model(x, w, b):
    Sig = nn.Sigmoid()
    return Sig(x.mm(w) + b)

def lossfunc(ypre, ytar):
    CE = nn.BCEWithLogitsLoss()
    return CE(ypre, ytar)

#%% Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

#%% Parameter
dim = X_train.shape[1]
x = torch.from_numpy(X_train).float()
Y_train = np.reshape(Y_train, (X_train.shape[0], 1))
ytar = torch.from_numpy(Y_train)
w = torch.zeros((dim, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

lr = 0.1
ep = 100


train_loss = np.zeros([ep,1])
opt = optim.Adam([w, b], lr)
#%% Training
for t in range(ep):
    opt.zero_grad()
    ypre = model(x, w, b)
    loss = lossfunc(ypre, ytar)
    loss.backward()
    opt.step()
    eploss = loss.detach().numpy()
    if (t%10 == 0 or t == ep-1):
        print(str(t) + "_loss:" + str(eploss))
        
#%%Test
xt = torch.from_numpy(X_test).float()
ytpre = np.round(model(xt, w, b).detach().numpy()).astype(np.int)

with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(ytpre):
        f.write('{},{}\n'.format(i, label))
