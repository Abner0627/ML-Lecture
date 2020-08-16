# Binary Classification

> data:  
[https://www.kaggle.com/c/ml2020spring-hw2](https://www.kaggle.com/c/ml2020spring-hw2)

> into slide:  
[https://docs.google.com/presentation/d/1dQVeHfIfUUWxMSg58frKBBeg2OD4N7SD0YP3LYMM7AA/edit#slide=id.g7e958d1737_0_6](https://docs.google.com/presentation/d/1dQVeHfIfUUWxMSg58frKBBeg2OD4N7SD0YP3LYMM7AA/edit#slide=id.g7e958d1737_0_6)

> sample code:  
[https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C](https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C)

註：資料前處理的部分皆參考sample code

## Purpose

根據受試者的40項個人資料進行分類，預測該受試者之年薪是否有高於50,000美元。
為二元分類的問題。

## Training Data

![https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled.png](https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled.png)

在training data的部分，row代表受試者的編號（共計54256位），
column則包含該受試者的基本資料，並標註該受試者之年薪是否高於50,000美元。

此處資料經過處理後變為54256 * 510的矩陣，其中510項columns就是其input。
此外data的詳細資料可從"train.csv"中得知，如上圖所示。

## Testing Data

![https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled%201.png](https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled%201.png)

testing data為另外27622位受試者的資料，與training data不同該筆資料並無data，
是此次作業需要預測的目標。

同樣testing data亦做過資料處理，為27622 * 510的矩陣。

## Read Data

首先使用kaggle API將dataset載下來。

```jsx
kaggle competitions download -c ml2020spring-hw2
```

Import package與進行事先處理過的data，X_train、Y_train與X_test三筆。

```python
#%% Import package and set path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
```

## Functions

定義在該訓練中會被用到的model與functions。

"_normalize"對資料進行正規化，並由使用者決定是否只對特定的column進行，
或是否輸出其mean與std。
另外為求data處理方式相同，此處testing data須使用training data的mean與std。
該函式的詳細說明可參見sample code。

"model"與"lossfunc"定義了訓練時的模型與損失函數，此處用pytorch進行實作。
前者為linear model並通過sigmoid function之後使其值介於0~1之間，如下示：
i為受試者編號，n為510項input。

![https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled%202.png](https://abner0627.github.io/ML-Lecture/02_Binary_Classification/img/Untitled%202.png)

"shuffle"在每次訓練迴圈之前會打亂X與Y的順序。但仍使每筆X的row皆對應到原先的Y，
即受試者與其對應的label是固定的。
同樣可參見sample code看詳細說明。

```python
#%% Functions
def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column] ,0).reshape(1, -1)
    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)    # 避免為0
    return X, X_mean, X_std

def model(x, w, b):
    Sig = nn.Sigmoid()
    return Sig(x.mm(w) + b)

def lossfunc(ypre, ytar):
    CE = nn.BCEWithLogitsLoss()    #二元分類用的cross entropy
    return CE(ypre, ytar)

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
```

## Normalization

對data做正規化。

```python
#%% Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
```

## Parameters

首先將Y_train拉成54256 * 1的矩陣，使其與model出來的預測值長度相符。
接著決定weight與bias的長度，分別為510 * 1與1 * 1。

設定learning rate (lr)；
然後決定要打亂X_train與Y_train的順序多少次，使資料順序對training的影響變小；
接著設定要訓練的次數 (ep)。

最終定義optimizer，此處選用SGD加快速度，其momentum為預設值0.9。

```python
#%% Parameter
Y_train = np.reshape(Y_train, (X_train.shape[0], 1))
dim = X_train.shape[1]

w = torch.zeros((dim, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

lr = 15
ep_sh = 70
ep = 50
mot = 0.9
opt = optim.SGD([w, b], lr, momentum = mot)
```

## Training

在每次進入training的迴圈時，皆打亂資料順序，計ep_sh = 70次。
並在進入迴圈後每10個epoch就print出其loss值，最終再計算training data的準確率 (acc)。

```python
#%% Training
for sh in range(ep_sh):
    X_train, Y_train = shuffle(X_train, Y_train)
   
    for t in range(ep):
        x = torch.from_numpy(X_train).float()
        ytar = torch.from_numpy(Y_train)
        
        opt.zero_grad()
        ypre = model(x, w, b)
        
        loss = lossfunc(ypre, ytar)
        loss.backward()
        opt.step()
        eploss = loss.detach().numpy() 
        if (t%10  == 0 or t == ep-1):
            print("shuffle_epoch: " + str(sh) + "\n" + str(t) + "_loss: " + str(eploss))

acc = 1 - np.mean(np.abs(ypre.detach().numpy() - ytar.detach().numpy()))  
print("accuracy:" + str(acc))
```

## Testing

讀取testing data後與訓練得到的w跟b進行預測，再對其結果四捨五入後轉成int。

最終將結果以檔名"output_logistic.csv"輸出並上傳至kaggle。

```python
#%% Testing
xt = torch.from_numpy(X_test).float()
ytpre = np.round(model(xt, w, b).detach().numpy()).astype(np.int)

#%% Output csv.
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(ytpre):
        f.write('{},{}\n'.format(i, label))
```

## Result

最終結果如下 (Accuracy)：
若為改善其精度，可考慮採用sample code的mini-batch方式進行訓練，除增加運算速度外，
另可增加參數更新的次數。
（將資料分成數個batch，每看過一個batch之後就更新一次參數，同時改變learning rate）
接著亦可從原始資料的詳細內容著手，對影響重大的feature分開處理（如年齡與工作年資），
稱特徵工程 (feature engineering)  

| Method          | Public  | Private |
|-----------------|---------|---------|
| Pytorch_SGD    | 0.88415 | 0.88132 |
| Strong_baseline | 0.89052 | 0.89102 |
| Simple_baseline | 0.88617 | 0.88675 |
