#!/usr/bin/env python
# coding: utf-8

# In[152]:


import os
import math
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.optim import Adam
np.set_printoptions(threshold=np.inf)


# In[153]:


def tround(num, dec=0):
    if (np.isnan(num)):
        return 0
    else:
        a = math.floor(float(num))
        if (num%1>=0.5):
            return(a+1)
        else:
            return(a)


# In[154]:


scoredf = pd.read_csv("../Time-Table-Annotation.csv")
AU = pd.read_csv("../OpenFace_200/A1/AU/002-A-L-AU.csv")
score = scoredf.loc[1:,["ID","Class Level"]]
score=score.dropna().values
train_index = np.array([1,2])
test_index = np.array([1,2])


# In[155]:


def Dataset_create(train_index, test_index, AU, score):
    X_train = np.empty([AU.T.shape[0],AU.T.shape[1]])
    X_test = np.empty([AU.T.shape[0],AU.T.shape[1]])
    count = []
    for filenum in train_index:
        if (''.join(glob.iglob('../OpenFace_200/**/'+score[filenum][0]+'*-AU.csv', recursive=True))==''):
            count.append(filenum)
        else:
            for filename in glob.iglob('../OpenFace_200/**/'+score[filenum][0]+'*-AU.csv', recursive=True):
                AU = pd.read_csv(filename).values
                if (X_train.shape[1]>AU.T.shape[1]):
                    X_train = X_train[:,0:AU.T.shape[1]] #reduces features to the minimum of all test data. Is this ok?
                X_train = np.dstack((X_train, AU.T[:,0:X_train.shape[1]]))
    train_index = train_index.tolist()
    for i in count:
        train_index.remove(i)
    count =[]
    for filenum in test_index:
        if (''.join(glob.iglob('../OpenFace_200/**/'+score[filenum][0]+'*-AU.csv', recursive=True))==''):
            count.append(filenum)
        else:
            for filename in glob.iglob('../OpenFace_200/**/'+score[filenum][0]+'*-AU.csv', recursive=True):
                AU = pd.read_csv(filename).values
                if (X_test.shape[1]>AU.T.shape[1]):
                    X_test = X_test[:,0:AU.T.shape[1]] #reduces features to the minimum of all test data. Is this ok?
                    X_train = X_train[:,0:AU.T.shape[1]] #reduces features to the minimum of all test data. Is this ok?
                X_test = np.dstack((X_test, AU.T[:,0:X_test.shape[1]]))
    test_index = test_index.tolist()
    for i in count:
        test_index.remove(i)
    y_test = np.array([tround(float(score[i][1])) for i in test_index])
    y_train = np.array([tround(float(score[i][1])) for i in train_index])
    X_train = X_train[:,:,1:]
    X_test = X_test[:,:X_train.shape[1],1:]
    X_train = X_train.transpose(2,0,1)
    #X_train = np.reshape(X_train[:,:,:], [-1, X_train.shape[1]*X_train.shape[-1]])
    X_test = X_test.transpose(2,0,1)
    #X_test = np.reshape(X_test[:,:,:], [-1, X_test.shape[1]*X_test.shape[-1]])
    return X_train, X_test, y_train, y_test


# In[156]:


def reshapeindices_flat(X_train, X_test):
    X_train = np.reshape(X_train, [-1, X_train.shape[1]*X_train.shape[-1]])
    X_test = np.reshape(X_test, [-1, X_test.shape[1]*X_test.shape[-1]])
    return X_train, X_test


# In[157]:


def reshapeindices_split(data,frames, axis):
    splitint = math.floor(data.shape[axis]/frames)
    split_indices = [frames*(i+1) for i in range(splitint-1)]
    data = data[:,:,:(splitint*5)]
    data = np.array_split(data, split_indices, axis=2)
    data=np.stack(data)
    return data


# In[158]:


def VerificationDataset(trainsize, featsize, samplelength, testsize): #This is for debug purposes, generates a similar dataset with label 1-6 and train data 1-6.
    x = np.floor(np.arange(trainsize)/(trainsize/6))
    c = np.ones((featsize,samplelength))
    X_train = x[..., None, None] * c[None, :, :]
    y = np.floor(np.arange(testsize)/(testsize/6))
    X_test = y[..., None, None] * c[None, :, :]
    y_train = x
    y_test = y
    return X_train, X_test, y_train, y_test


# In[159]:


class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()
        self.input_layer = nn.Linear(inputDim, hiddenDim)
        self.rnn = nn.LSTM(input_size = hiddenDim,
                            hidden_size = hiddenDim,
                            batch_first = True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
        #self.softmax= nn.Softmax(dim=1)
    
    def forward(self, inputs, hidden0=None):
        #print(inputs.shape)
        #inputs = inputs.permute(1,0,2)
        #print(inputs.shape)
        output = self.input_layer(inputs) #行列サイズ対処
        #print(output.shape)
        output, (hidden, cell) = self.rnn(output, hidden0) #LSTM層
        output = self.output_layer(output[:, -1, :]) #全結合層
        #output = self.softmax(output)
        
        return output


# In[ ]:





# In[164]:


kf = KFold(n_splits = 10, shuffle = True)
for train_index, test_index in kf.split(score):
    X_train, X_test, y_train, y_test = Dataset_create(train_index, test_index, AU, score)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    break


# In[188]:


#X_train, X_test, y_train, y_test = VerificationDataset(500, 18, 20, 50)


# In[189]:


X_train = reshapeindices_split(X_train, 5, 2)
X_test = reshapeindices_split(X_test, 5, 2)
X_train = X_train.transpose(2,3,0,1)
X_test = X_test.transpose(2,3,0,1)
tileamount = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3],1)
X_train = X_train.transpose(2,3,1,0)
X_test = X_test.transpose(2,3,1,0)
y_traintensor = np.tile(y_train, tileamount)
y_traintensor = np.repeat(y_traintensor[None, :], X_train.shape[2], axis=0)
y_testtensor = np.tile(y_test, tileamount)
y_testtensor = np.repeat(y_testtensor[None, :], X_train.shape[2], axis=0)
X_train=X_train[:,:,:,1:]
X_test = X_test[:,:,:,1:]


# In[190]:


print(y_traintensor.shape, y_testtensor.shape)


# In[196]:


training_size = X_train.shape[0] #traning dataのデータ数
epochs_num = 100 #traningのepoch回数
hidden_size = 50 #LSTMの隠れ層の次元数

model = Predictor(X_train.shape[3], hidden_size, 7) #modelの宣言

criterion = nn.CrossEntropyLoss() #評価関数の宣言
optimizer = Adam(model.parameters(), lr=0.001) #最適化関数の宣言


# In[197]:


running_losscount = []
training_accuracycount = []
for epoch in range(epochs_num):
    dataoutput = []
    dataanswer = []
    running_loss = 0.0
    training_accuracy = 0.0
    for i in range(training_size): #X_train = (training_size, batch_size, sample_size, feat_size)
        optimizer.zero_grad()
        data = torch.tensor([X_train[i][0]]).float()
        label = torch.tensor([y_traintensor[0][i]]).long().T
        #print(data)
        #print(label)
        output = model(data.float())
        #print(output.shape)
        #print(label.shape)
        #print(output)
        #print(label)
        #print(torch.min(output), torch.max(output))
        #print(torch.min(label), torch.max(label))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #print(output.shape)
        output = torch.argmax(output, dim=1)
        #print("output " + str(output.data.shape))
        #print(output.data)
        #print("answer " + str(label.data.shape))
        #print(label.data)
        #print(np.sum((np.abs((output.data - label.data).numpy()) < 0.1))/len(output.data))
        training_accuracy += np.sum((np.abs((output.data - label.data).numpy()) < 0.1))/len(output.data)
        dataoutput.append(int(output.data[0]))
        dataanswer.append(int(label.data[0]))
        if (((i+1)%10000)==0):
            print(str(i+1)+"epochs has passed. It's working, don't worry.")
    training_accuracy /= training_size
    running_loss /= (training_size*len(output.data))
    print('%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))
    print(dataoutput[::20000])
    print(dataanswer[::20000])
    running_losscount.append(running_loss)
    training_accuracycount.append(training_accuracy)
    test_accuracy = 0.0
    test_size = int(y_testtensor.shape[1])
    for i in range(test_size):
        data, label = torch.tensor([X_test[i][0]]).float(), torch.tensor([y_testtensor[0][i]]).long().T
        output = model(data.float(), None)
        output = torch.argmax(output, dim=1)
        test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)/len(output.data)
    
    print(test_accuracy)
    test_accuracy /= test_size

    print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy, test_accuracy))


# In[ ]:




