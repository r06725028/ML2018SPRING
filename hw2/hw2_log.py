import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import csv


#參數
max_iter = 1000
lr = 0.1
batch_size = 32
valid_num = 0.1

def read_datax(file):
    data = []
    with open(file, "r", encoding="big5") as f:
        for line in list(csv.reader(f))[1:]:
            data.append([float(x) for x in line])
    data = np.array(data)

    return data

def read_datay(file):
    data = []
    with open(file, "r", encoding="big5") as f:
        for line in list(csv.reader(f)):
            data.append([float(x) for x in line])
    data = np.array(data)

    return data

def shuffle_data(train_x, train_y):
    index = np.arange(len(train_x))
    np.random.shuffle(index)
    
    train_x = train_x[index]
    train_y = train_y[index]
    
    return train_x[index], train_y[index]


def normalize_data(train_x, test_x):
    #用train_x+test_x來算mean,std????????
    mean = np.mean(train_x,axis=0)
    std = np.std(train_x,axis=0)
    
    train_x = (train_x-mean)/std
    test_x = (test_x-mean)/std
    
    return train_x,test_x

def add_feature(x):
    for i in range(x.shape[1]):
        if max(x[:,[i]]) > 1:
            x1 = x[:, [i]] ** 3
            x2 = x[:,[i]] ** 0.5
            x = np.concatenate((x, x1,x2), axis = 1)
    return x

def valid_data(train_x, train_y, num):
    m = len(train_x)
    n = int(m*num)
    new_m = m - n  

    new_train_x = train_x[0:new_m]
    new_train_y = train_y[0:new_m] 
    valid_x = train_x[new_m:]
    valid_y = train_y[new_m:]
    
    return new_train_x,new_train_y,valid_x,valid_y

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    # 用clip把數值限制於給定的大小值，避免溢位？
    return np.clip(res, 1e-8, 1-(1e-8))

def count_valid(w, b, valid_x, valid_y):
    m = len(valid_x)
    z = (np.dot(valid_x, np.transpose(w)) + b)
    y = sigmoid(z)

    #用四捨五入決定零或一
    y_pred = np.around(y)
    #把維度為一的去掉，變成(16385,)
    result = (np.squeeze(valid_y) == y_pred)
    #print(result)
    
    cross_entropy = -1 * (np.dot(np.squeeze(valid_y), np.log(y)) + np.dot((1 - np.squeeze(valid_y)), np.log(1 - y)))
    print('Validation acc = %f' % (float(result.sum()) / m))
    print('Validation loss = %f' % (float(cross_entropy) / m))
    

def train(train_x,train_y):

    train_x,train_y,valid_x,valid_y = valid_data(train_x,train_y,valid_num)
    
    w = np.zeros((train_x.shape[1],))
    b = np.zeros((1,))
    
    m = len(train_x)
    step_num = int(floor(m / batch_size))
    
    total_loss = 0.0
    for epoch in range(1, max_iter):
        train_x, train_y = shuffle_data(train_x, train_y)

        for idx in range(step_num):
            batch_x = train_x[idx*batch_size:(idx+1)*batch_size]
            batch_y = train_y[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(batch_x, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(batch_y), np.log(y)) + np.dot((1 - np.squeeze(batch_y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_grad = np.mean(-1 * batch_x * (np.squeeze(batch_y) - y).reshape((batch_size,1)), axis=0)
            b_grad = np.mean(-1 * (np.squeeze(batch_y) - y))

            w = w - lr * w_grad
            b = b - lr * b_grad

        if (epoch % 50) == 0:
            """
            z = (np.dot(train_x, np.transpose(w)) + b)
            y = sigmoid(z)
            #用四捨五入決定零或一
            y_pred = np.around(y)
            #把維度為一的去掉，變成(16385,)
            result = (np.squeeze(train_y) == y_pred)
            print(result)
            """
            print('=====epoch %d=====' % epoch)
            print('epoch avg loss = %f' % (total_loss / (float(50) * m)))
            #print('epoch acc = %f' % (float(result.sum()) / m))
            
            count_valid(w, b, valid_x, valid_y)
            
            total_loss = 0.0

    with open ('log_w.npy','wb') as f:
        np.save(f,w)
    with open ('log_b.npy','wb') as f:
        np.save(f,b)

def output(test_x):
    m = len(test_x)

    w = np.load('log_w.npy')
    b = np.load('log_b.npy')

    z = (np.dot(test_x, np.transpose(w)) + b)
    y = sigmoid(z)
    y_pred = np.around(y)

    with open(sys.argv[6], 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_pred):
            f.write('%d,%d\n' %(i+1, v))


def main():
    train_x = add_feature(read_datax(sys.argv[3]))
    train_y = read_datay(sys.argv[4])
    test_x = add_feature(read_datax(sys.argv[5]))

    train_x,test_x = normalize_data(train_x,test_x)
    train_x,train_y = shuffle_data(train_x,train_y)

    print(train_x.shape) 
    print(test_x.shape)
    print(train_y.shape)
    

    train(train_x,train_y)
    output(test_x)

train_x = add_feature(read_datax(sys.argv[3]))
test_x = add_feature(read_datax(sys.argv[5]))

train_x,test_x = normalize_data(train_x,test_x)

output(test_x)
