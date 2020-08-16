# Regression

> data:  
[https://www.kaggle.com/c/ml2020spring-hw1](https://www.kaggle.com/c/ml2020spring-hw1)

> into slide:  
[https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_10](https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_10)

> sample code:  
[https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=U7RiAkkjCc6l](https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C#scrollTo=U7RiAkkjCc6l)

註：資料前處理的部分皆參考sample code

## Purpose

根據前9個小時的feature（共18項觀測指標），預測第10個小時的PM2.5數值。

## Data

主要分train.csv與test.csv，前者是豐原站每個月的前 20 天所有資料；<br />
後者則是從豐原站剩下的資料中取樣出來。

## Training Data

![https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled.png](https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled.png)

在training data的部分，row為每月前20天的18項觀測指標（計12 * 20 * 18 = 4320項）；<br />
column為其每天24小時量測的數值（計24項）。

因此training data為4320 * 24的矩陣。

## Testing Data

![https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%201.png](https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%201.png)

testing data為剩下的資料sample出每筆連續10小時，共240筆的觀測數值（240 * 18 = 4320項）。<br />
而第10小時的PM2.5數值當作預測的答案。

因此testing data為4320 * 9的矩陣。

## Read Data

首先使用kaggle API將dataset載下來。

```jsx
kaggle competitions download -c ml2020spring-hw1
```

Import package與data，並將training data中含有文字的部分去除以及NaN的部分設為0。

```python
#%% Import package and data
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

raw_data = pd.read_csv('.data/train.csv',encoding='big5')
raw_data = raw_data.to_numpy()    # 將dataframe轉為array
data = raw_data[:,3:]
data[data=="NR"] = 0
```

接著把每個月20天的資料以18項量測指標為row重新排列，如下圖示，最終得到18 * 480的矩陣。<br />
而另將1~12月的所有18 * 480的矩陣以dictionary的方式儲存 (month_data)。

```python
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample
```

![https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%202.png](https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%202.png)

## Parameters and model

將每個月矩陣中480個columns，每隔1小時以10小時為一組（共471組，多餘的捨去）進行分類。<br />
前9小時為input x的值；第10小時為training target y的值。

```python
x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue
                # 跳過第20天第15小時(不含)以後的data(超出471)
            x[month * 471 + day * 24 + hour,:] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1)
            y[month * 471 + day * 24 + hour,0] = month_data[month][9 ,day * 24 + hour + 9]
```

![https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%203.png](https://abner0627.github.io/ML-Lecture/01_Regression/img/Untitled%203.png)

對x取normalize 。

```python
mean_x = np.mean(x, axis = 0)
std_x = np.std(x, axis = 0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
```

定義model與loss functions (RMSE)並設定參數以及optimizer (Adam)，此處以pytorch進行實作。

```python
def model(x):
    return x.mm(w) + b

def lossfunc(ypre, ytar):
    MSE = nn.MSELoss()
    return torch.sqrt(MSE(ypre, ytar))

dim = 18 * 9
w = torch.zeros((dim, 1), requires_grad=True)    # weight
b = torch.zeros(1, requires_grad=True)    # bias
lr = 0.1    # learning rate
ep = 1000    # epoch，訓練次數
x_in = torch.from_numpy(x).float()
ytar = torch.from_numpy(y)
optimizer = optim.Adam([w, b], lr)
```

## Training

進行training並將得出的weight與bias轉成numpy array，作為待會testing之用。

```python
for t in range(ep):
    optimizer.zero_grad()    # initailize optimizer
    ypre = model(x_in).double()    # training prediction
    loss = lossfunc(ypre, ytar)    # calculate loss function
    loss.backward()    # backpropagation
    optimizer.step()    # update all parameters (e.g. weight, bias, etc.)
    if (t%100 == 0 or t == ep - 1):
        print(str(t) + ":" + str(loss.detach().numpy()))
		# 每100個epoch print一次loss的值

w_array = w.detach().numpy()
b_array = b.detach().numpy()
```

## Testing

讀testing data並同樣排成以18個量測指標為row的矩陣。

```python
testdata = pd.read_csv('filepath/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]    # 去除文字
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)

for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
```

使用訓練得到的weight與bias行預測，將結果存成 .csv上傳至kaggle。

```python
ans_y = np.dot(test_x, w_array) + b_array
np.savetxt("ansy.csv", ans_y, delimiter=",")
```

最終成果如下 (RMSE)：<br />
可將training data分作training set與validation set，以便在上傳kaggle前就能評估model的效能，<br />
藉此改善其在private score的精度。  

| Method          | Public  | Private |
|-----------------|---------|---------|
| Pytorch_Adam    | 5.44949 | 7.46378 |
| Strong_baseline | 7.14231 | 7.14231 |
| Simple_baseline | 8.73773 | 8.73773 |
