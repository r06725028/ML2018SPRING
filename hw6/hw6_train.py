import math
import pandas as pd
import numpy as np

from keras.layers import Embedding, Reshape, Input, Dot, Add, Merge, Dropout, Dense
from keras.models import Model, Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Concatenate

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import argparse,os,sys

from sklearn.utils import shuffle

#global參數
train_path = 'train.csv'
latent_dim=150#120#90
dnn = True
normalize = False
save_path = 'aaa/'

def get_model(n_users, n_items, latent_dim=120):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	user_vec = Dropout(0.5)(user_vec)

	item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec) 
	item_vec = Dropout(0.5)(item_vec)

	user_bias = Embedding(n_users, 1, embeddings_initializer='random_normal')(user_input)
	user_bias = Flatten()(user_bias)

	item_bias = Embedding(n_items, 1, embeddings_initializer='random_normal')(item_input)
	item_bias = Flatten()(item_bias)

	r_hat = Dot(axes=1)([user_vec, item_vec])
	r_hat = Add()([r_hat, user_bias, item_bias])

	model = Model([user_input,item_input], r_hat)
	#model.compile(loss='mse', optimizer='sgd')
	model.compile(loss='mse', optimizer='adam')

	return model

def nn_model(n_users, n_items, latent_dim=120):
	user_input = Input(shape=[1])
	item_input = Input(shape=[1])
	
	user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	user_vec = Dropout(0.5)(user_vec)

	item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input)
	item_vec = Flatten()(item_vec) 
	item_vec = Dropout(0.5)(item_vec)

	merge_vec = Concatenate()([user_vec, item_vec])

	#hidden = Dense(150, activation='relu')(merge_vec)
	hidden = Dense(150, activation='linear')(merge_vec)
	hidden = LeakyReLU(alpha=0.001)(hidden)
	hidden = Dropout(0.5)(hidden)
	#hidden = Dense(50, activation='relu')(hidden)
	output = Dense(1,activation='linear')(hidden)
	output = LeakyReLU(alpha=0.001)(output)

	model = Model([user_input,item_input], output)
	#model.compile(loss='mse', optimizer='sgd')
	model.compile(loss='mse', optimizer='adam')
	model.summary()

	return model

def draw(x,y):
	from matplotlib import pyplot as pyplot
	from tsne import bh_sne
	plt.switch_backend('agg') 

	y = np.array(y)
	x = np.array(x, dtype=np.float64)

	#perform t-SNE embedding
	vis_data = bh_sne(x)

	#plot the result
	vis_x = vis_data[:, 0]
	vis_y = vis_data[:, 1]

	cm = plt.cm.get_camp('RdYlBu')
	sc = plt.scatter(vis_x,vis_y, c=y, cmap=cm)
	plt.colorbar(sc)
	plt.title(title)

	plt.legend(loc="upper right")
	plt.show()

	path = 'aa.jpg'
	plt.savefig(path)	

def get_embedding(model):
	#get embedding
	user_emb = np.array(model.layers[2].get_weights()).squeeze()
	print('user embedding shape:', user_emb.shape)

	movie_emb = np.array(model.layers[3].get_weights()).squeeze()
	print('movie embedding shape:', user_emb.shape)

	np.save('user_emb.npy', user_emb)
	np.save('movie_emb.npy', movie_emb)
	return None

#讀檔並作shffule
train_df = pd.read_csv(train_path)
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df = shuffle(train_df)
print('read data ok')

#找出最大的id，即為數量
user_num = train_df['UserID'].unique().max()
movie_num = train_df['MovieID'].unique().max()

if dnn:
	model = nn_model(user_num, movie_num,latent_dim)
else:
	model = get_model(user_num, movie_num,latent_dim)


earlystopping = EarlyStopping(monitor='val_loss', patience = 2, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath=save_path+'{epoch:02d}.h5', 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     mode='max' )
csv_logger = CSVLogger(save_dir+'train.csv', separator=',', append=False)

train_x = [train_df['UserID'].values - 1, train_df['MovieID'].values - 1]
train_y = train_df['Rating'].values
print('process data ok')

model.fit(train_x,train_y , epochs=100, batch_size=1024, validation_split=0.2, verbose=1, callbacks=[earlystopping,checkpoint,csv_logger])


