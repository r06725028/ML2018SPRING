import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import csv


#參數
#max_iter = 1000
#lr = 0.1
#batch_size = 32
valid_num = 0.1
save_dir = './gen_model/'
if not os.path.exists(save_dir):
        os.mkdir(save_dir)


#輸出精準度
np.set_printoptions(precision = 5, suppress = True)
#避免underflow
np.seterr(divide='ignore', invalid='ignore')


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
    
    return train_x, train_y


def normalize_data(train_x, test_x):
    #用train_x+test_x來算mean,std????????
    mean = np.mean(train_x,axis=0)
    std = np.std(train_x,axis=0)
    
    train_x = (train_x-mean)/std
    test_x = (test_x-mean)/std
    
    return train_x,test_x

def add_feature(x):
    """
    for i in range(x.shape[1]):
        if max(x[:,[i]]) > 1:
            x1 = x[:, [i]] ** 3
            x2 = x[:,[i]] ** 0.5
            x = np.concatenate((x, x1,x2), axis = 1)
    """
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

def count_valid(valid_x, valid_y, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = valid_x.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_pred = np.around(y)
    result = (np.squeeze(valid_y) == y_pred)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))

def train(train_x,train_y):

    train_x,train_y,valid_x,valid_y = valid_data(train_x,train_y,valid_num)
    m = len(train_x)
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((train_x.shape[1],))
    mu2 = np.zeros((train_x.shape[1],))
    for i in range(m):
        if train_y[i] == 1:
            mu1 += train_x[i]
            cnt1 += 1
        else:
            mu2 += train_x[i]
            cnt2 += 1
    
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((train_x.shape[1],train_x.shape[1]))
    sigma2 = np.zeros((train_x.shape[1],train_x.shape[1]))
    
    for i in range(m):
        if train_y[i] == 1:
            sigma1 += np.dot(np.transpose([train_x[i] - mu1]), [(train_x[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([train_x[i] - mu2]), [(train_x[i] - mu2)])
    
    sigma1 /= cnt1
    sigma2 /= cnt2
    
    shared_sigma = (float(cnt1) / m) * sigma1 + (float(cnt2) / m) * sigma2
    
    N1 = cnt1
    N2 = cnt2

    
    param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param_dict):
        np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
    print('=====Validating=====')
    count_valid(valid_x, valid_y, mu1, mu2, shared_sigma, N1, N2)


def output(test_x):
    m = len(test_x)

    mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    N2 = np.loadtxt(os.path.join(save_dir, 'N2'))
    
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = test_x.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
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

#main()

train_x = add_feature(read_datax(sys.argv[3]))
test_x = add_feature(read_datax(sys.argv[5]))

train_x,test_x = normalize_data(train_x,test_x)

output(test_x)

