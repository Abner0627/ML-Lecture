# CNN

> data:  
[https://www.kaggle.com/c/ml2020spring-hw3](https://www.kaggle.com/c/ml2020spring-hw3)

> into slide:  
[https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p1](https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p1)

> sample code:  
[https://colab.research.google.com/drive/16a3G7Hh8Pv1X1PhZAUBEnZEkXThzDeHJ](https://colab.research.google.com/drive/16a3G7Hh8Pv1X1PhZAUBEnZEkXThzDeHJ)

註：下列內容皆為sample code擷取過來的，我嘗試用自己的方式解釋其運作方式。
　　詳細內容請參閱上述連結。

## Purpose

根據圖片中食物種類利用CNN將其分為下列11項類別。
1.    Bread
2.    Dairy product
3.    Dessert
4.    Egg
5.    Fried food
6.    Meat
7.    Noodles/Pasta
8.    Rice
9.    Seafood
10.  Soup
11.  Vegetable/Fruit

## Training Data

![https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled.png](https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled.png)

training data為以 [LABEL]_[NUMBERING] 的形式命名的圖片，共9866張。

## Validation Data

![https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled%201.png](https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled%201.png)

validation data與training data以相同方式命名，主要用來評估該model的表現，數量共有3430張。

## Testing Data

![https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled%202.png](https://abner0627.github.io/ML-Lecture/03_CNN/img/Untitled%202.png)

testing data不具label，僅是單純將圖片以收集的順序命名，總計3347張。
因此machine任務就是將該3347張食物圖片分作11項類別。

## Read Data

同樣使用kaggle API將dataset載下來。

```jsx
kaggle competitions download -c ml2020spring-hw3
```

Import package，並定義readfile function使用OpenCV (cv2)套件讀取照片並存於numpy array中。

```python
#%% Import package
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

#%% OpenCV
def readfile(path, label):
# label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x
```

接著使用readfile function將training set, validation set與testing set的圖片讀進來。

```python
workspace_dir = './food-11'
# 讀取圖片路徑
print("Reading data")

train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))

val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

test_x = readfile(os.path.join(workspace_dir, "testing"), False)
# 不具 label，因此只需要 return x
print("Size of Testing data = {}".format(len(test_x)))
```

## Pack as Dataset

為將現有的training data性能發揮到最大，盡可能彌補資料量較少造成over-fitting的情形。
詳細參閱：[https://reurl.cc/0OzrX6](https://reurl.cc/0OzrX6)

基於該需求，data augmentation僅需使用在training上即可。

```python
train_transform = transforms.Compose([
    transforms.ToPILImage(),
		# 將numpy array轉為PIL以便後續進行影像處理
    transforms.RandomHorizontalFlip(), 
		# 隨機將圖片水平翻轉
    transforms.RandomRotation(15),
		# 隨機旋轉圖片
    transforms.ToTensor(), 
		# 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),                                    
    transforms.ToTensor(),
])
```

定義ImgDataset function，並將training set與validation set打包為batch size為128的dataset。

在__init__中匯入dataset (x)與其label (y)，此外若data有label存在 (如training/validation set)，則將y變為LongTensor的形式；最終再定義使用上述的transform流程。

__len__定義該dataset的大小；__getitem__則定義當程式取值時，dataset應該要怎麼回傳資料。
兩者為DataLoader函式在enumerate Dataset時會使用到，因此為必要項目。

參見：[https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
　　　[https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

```python
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
						# label is required to be a LongTensor

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 128
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# trainning 時為減少 data 順序影響 model 的程度，使用 shuffle 打亂其順序
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
```

## Model

定義CNN model並設定其forward()。

在nn.Conv2d()中宣告input與output的dimension、convolution中kernel的大小、每次kernel的步進大小 (stride)，以及zero-padding在input兩邊補0的column數量 (padding)。
參見：[https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

nn.BatchNorm2d()表示在該層layer中是否使用Batch Normalization，藉此確保每層layer的input保持相同分佈。
參見：[https://zhuanlan.zhihu.com/p/88347589](https://zhuanlan.zhihu.com/p/88347589)

nn.MaxPool2d()中宣告需要做max pooling的範圍 (kernel_size, n*n)，以及zero-padding的數量 (padding)。
參見：[https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html](https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html)

最終通過linear layer預測其分類，nn.Linear()分別設定input與output feature大小。
參見：[https://pytorch.org/docs/stable/generated/torch.nn.Linear.html](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

另外在pytorch終須自行設定network的forward方向。

```python
#%% Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        # torch.nn.Linear(in_features, out_features, bias: bool = True)
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
```

## Training

設定model以及是否使用GPU加速；
此處為多元分類的問題，因此loss function為Cross Entropy；
Optimizer選用adam，且learning rate為0.001；
最終設定training epoch為30。

```python
#%% Training
model = Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 30
```

以epoch為30訓練model，並用train_acc與val_acc除以其set長度，
評估該epoch中的準確率，loss值同理。

另外用model.train()與model.eval()切換training與validation set，
區別其是否使用back propagation。

```python
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() 
		# 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):    
				# idx = i, item = data
        optimizer.zero_grad() 
				# 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) 
				# 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) 
				# 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() 
				# 利用 back propagation 算出每個參數的 gradient
        optimizer.step() 
				# 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
				# 累計 training 的準確率 (accuracy)
        train_loss += batch_loss.item()
				# 累計 training 的loss 值
    
    model.eval()
		# 不需要 backprop: 暫時不追踪 network 參數中的 gradient
    with torch.no_grad():    
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
				# 將結果 print 出來
        # %03d: 表示將整數用0補足到3位
        # %2.2f: 保留2位整數及小數點後2位
```

之後調整model參數並評估其性能，確認該model在當下參數表現最好後，
再concatenate training與validation set重複訓練一遍，提升其性能。

```python
#%% Training set & validation set train together         
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y, train_transform)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)
# 同樣打亂該 dataset 的順序

model_best = Classifier().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))
		# 同樣將結果 print 出來
```

## Testing

同樣將testing set打包為dataset，並預測其label。
最終將結果寫入csv並上傳kaggle。

```python
#%% Testing
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
```
