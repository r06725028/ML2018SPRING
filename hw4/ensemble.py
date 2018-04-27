import numpy as np
import pandas as pd
import csv
from random import shuffle
import sys
import os

import tensorflow 
import keras
import string

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam#不固定的梯度下降
from keras.backend import argmax
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#global參數
epochs = 250
batch_size = 128
validation_split = 0.2
shuffle = True

"""
#新增資料夾
dir_path = './myckpt16/'
if not os.path.isdir(dir_path):
  os.mkdir(dir_path)
"""

def readData(file):
	label, data = [], []
	
	with open(file, "r") as f:
		for line in list(csv.reader(f))[1:]:	
			label.append([int(line[0])])
			data.append([float(x)/255 for x in line[1].split()])
			
	data = np.array(data).reshape((-1, 48, 48, 1))

	return data
	
def normal(data):
	print(data.shape)
	
	mean = np.load(open('mean.npy','rb'))
	std = np.load(open('std.npy','rb'))
	print(((data-mean)/std).shape)
	return  (data-mean)/std

def output():
	test_x = normal(readData('test.csv'))

	#model_list = []
	sum_ = 0
	for i in range(1,4):
		#model.append(load_model(sys.argv[i]))
		model = load_model('model_'+str(i)+'.h5')
		sum_ += model.predict(test_x)
		
	result = np.argmax(sum_,axis=-1)

	with open('ensemble_op.csv', 'w') as f:
		f.write('id,label\n')
		for i, v in  enumerate(list(result)):
			f.write('%d,%d\n' %(i, v))

output()
