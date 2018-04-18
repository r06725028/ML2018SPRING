import numpy as np
import pandas as pd
import csv
from random import shuffle
import sys
import os

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


#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout
#from tensorflow.python.keras.regularizers import l2

from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#import pickle as pkl


#global參數
epochs = 250
batch_size = 128
validation_split = 0.2
validation_num = 5000
shuffle = True
isTrain = True
num_classes = 7
dim = 48

datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, \
	zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)


#新增資料夾
dir_path = './myckpt16/'
if not os.path.isdir(dir_path):
  os.mkdir(dir_path)



def readData(file):
	label, data = [], []
	
	with open(file, "r") as f:
		for line in list(csv.reader(f))[1:]:	
			label.append([int(line[0])])
			data.append([float(x)/255 for x in line[1].split()])
			
	data = np.array(data).reshape((-1, 48, 48, 1))

	return data, np_utils.to_categorical(np.array(label),num_classes=7)
	
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
	
def normal(data):
	try:
		mean = np.load(open('mean.npy','rb'))
		std = np.load(open('std.npy','rb'))
	except:
		mean = np.mean(data,axis=0)
		std = np.std(data,axis=0)
		
		with open('mean.npy','wb') as f:
			np.save(f,mean)
		with open('std.npy','wb') as f:
			np.save(f,std)
	
	return  (data-mean)/std

def train(x,y):
	model = Sequential()#sigmoid
	
	model.add(Dense(128,kernel_regularizer=l2(0.0),input_shape=(x.shape[1],)))#activation='sigmoid'
	model.add(BatchNormalization())
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	
	model.add(Dense(64,kernel_regularizer=l2(0.0)))#, activation='sigmoid'
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	
	model.add(Dense(2))#,activation='softmax'
	model.add(Activation('softmax'))

	model.add(Dropout(0.5))
	model.add(Dense(1, activation='softmax',kernel_regularizer=l2(0.1),input_shape=(x.shape[1],)))
	model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
	
	hist = model.fit(x,y,epochs=50, batch_size=32,verbose=1,validation_split=0.2, shuffle=True)
	class_weight={0:1.0,1:1.25}

	return model,hist

def myCnn():
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(48,48,1),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(512, kernel_size=(3, 3),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Conv2D(512, kernel_size=(3, 3),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))


	model.add(Conv2D(256, kernel_size=(3, 3),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(128, kernel_size=(3, 3),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	"""
	model.add(Conv2D(256, kernel_size=(3, 3),padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))

	model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
	model.add(LeakyReLU(alpha=0.05))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.5))
	"""

	model.add(Flatten())

	model.add(Dense(512, kernel_regularizer=l2(0), kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	
	model.add(Dense(512, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
	
	return model

def trainCnn():
	
	train_x, train_y = readData(sys.argv[1])
	train_x = normal(train_x)

	X_train, X_valid = train_x[:-5000], train_x[-5000:]
	Y_train, Y_valid = train_y[:-5000], train_y[-5000:]
	
	model = myCnn()
	
	callbacks = []
	callbacks.append(ModelCheckpoint(dir_path+'model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
	
	model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
	print(train_x.shape)
	#hist = model.fit(train_x,train_y,epochs=epochs, batch_size=batch_size,verbose=1,validation_split=validation_split, shuffle=shuffle, callbacks=callbacks)

	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), \
		steps_per_epoch=5*len(train_x)//batch_size, epochs=epochs,\
		validation_data=(X_valid, Y_valid), callbacks=callbacks)


	#model.save(sys.argv[2])
	
#with tf.device(/gpu:0):
trainCnn()



	






