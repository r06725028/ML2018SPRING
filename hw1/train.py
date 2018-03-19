import numpy as np
import pandas as pd
import csv
import sys
import math
#import random
#from tqdm import tqdm
#import pickle as pkl#改用np.save為npy檔
#from sklearn import linear_model

np.random.seed(123)
###############參數#################
lr = 1e-1
max_epoch = 50000

feature_list = ['AMB_TEMP', 'CH4', 'CO', 'NHMC', 'NO', 'NO2', 'NOx'\
	,'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC'\
	, 'WIND_SPEED', 'WS_HR']
####一、訓練資料處理
def pre_traindata(path):
	##1.讀csv檔為dataframe
	#train_df = pd.read_csv('./train.csv',encoding='utf-8')
	train_df = pd.read_csv(path,encoding='big5')
	assert len(train_df)==18*20*12,'read_csv error!!!'#18個觀測值＊每月20天＊一年12個月
	##2.只取出某一欄
	#print(train_df[train_df.columns[2]=='PM2.5'])#KeyError: False
	train_feature = {}
	for feature in feature_list:
		train_feature[feature] = train_df[train_df[train_df.columns[2]]==feature]
	#assert len(train_df)==20*12,'only pm2.5 error!!!'
	##2.移除前三欄
	print(len(train_df.columns))
	#DF.drop([DF.columns[[0,1, 3]]], axis=1,inplace=True) 
	train_df = train_df.drop(list(train_df.columns)[0:3],axis=1)
	print(len(train_df.columns))
	assert len(train_df.columns)==24,'columns num error!!!'#一天24小時
	##3.處理遺失值
	#train_df = train_df.fillna({"":0.0})
	train_df = train_df.replace(to_replace='NR', value=0.0)
	##4.分出x和y
	train_x = []
	train_y = []

	feature_no = 0
	one_x = {}
	for idx,row in enumerate(train_df.iterrows()):
		feature_no+=1
		ob_list = list(row[1])
		assert len(ob_list)==24,'len ob_list error!!!'#一天24小時

		if feature_no == 10:
			for i in range(24):
				if (idx+9)<24:
					one_x[feature_n].append(ob_list[i:i+9])
					train_y.append(float(ob_list[idx+9]))
		else:
			for i in range(24):
				if (idx+9)<24:
					one_x.append(ob_list[i:i+9])
			if feature_no == 18:
				train_x.append(np.array(one_x)).astype(np.float)
				feature_no = 0
				one_x = []
	##5.轉成矩陣
	x_ma = np.array(train_x)
	y_ma = np.array(train_y).astype(np.float)
	print('x shape = ',x_ma.shape)
	print('y shape = ',y_ma.shape)
	##6.存成pkl
	"""
	with open('all_x.pkl','wb') as f:
		pkl.dump(x_ma,f)
	with open('all_y.pkl','wb') as f:
		pkl.dump(y_ma,f)
	"""
	return x_ma,y_ma

def pre_testdata(path):
	test = []
	test_x = []
	test_df = pd.read_csv(path, header=None, encoding='big5')	
	#assert len(test_df)==18*260,'read_csv error!!!'#18個觀測值＊每月20天＊一年12個月
	##2.只取出pm2.5那欄
	#print(train_df[train_df.columns[2]=='PM2.5'])#KeyError: False
	#test_df = test_df[test_df[test_df.columns[1]]=='PM2.5']
	#train_df = train_df[train_df[train_df.columns[1]=='PM2.5']]
	#assert len(test_df)==260,'only pm2.5 error!!!'
	##2.移除前三欄
	#print(len(test_df.columns))
	#DF.drop([DF.columns[[0,1, 3]]], axis=1,inplace=True) 
	test_df = test_df.drop(list(test_df.columns)[0:2],axis=1)
	#print(len(test_df.columns))
	#assert len(test_df.columns)==9,'columns num error!!!'#一天24小時
	##3.處理遺失值
	#train_df = train_df.fillna({"":0.0})
	test_df = test_df.replace(to_replace='NR', value=0.0)
	##4.分出x
	for row in test_df.iterrows():
		pm25_list = list(row[1])
		assert len(pm25_list)==9,'len pm25_list error!!!'#一天24小時
		test_x.append(np.array(pm25_list))#.astype(np.float)	
	##5.轉成矩陣
	test_x_ma = np.array(test_x).astype(np.float)
	#print('test_x shape = ',test_x_ma.shape)
	with open(path, 'rb') as f:
		test_data = f.read()
		test_data = test_data.splitlines()
		i = 0
		for line in test_data:
			line = [x.replace("\'", "") for x in str(line).split(',')[2:]]
			if i % 18 == 10:
				line = [x.replace("NR", "0") for x in line]
			line = [float(x) for x in line]
			test.append(line)
			i += 1
	##6.存成npy
	with open('test_x.npy','wb') as f:
		np.save(f,test_x_ma)

	return test

def ten_fold(x_ma,y_ma):
	assert x_ma.shape[0] == y_ma.shape[0] , 'train x,y row not consist'
	#切出1/10
	vaild_part = x_ma.shape[0]/10 
	
	#打亂index順序
	index = list(range(x_ma.shape[0]))
	index = random.suffle(index)

	#存放結果
	x_va_list = []
	y_va_list = []
	x_ma_list = []
	y_ma_list = []

	for i in range(10):
		#切出1/10
		vaild_index = index[0+i*vaild_part:vaild_part+i*vaild_part]
		other_index = index-vaild_index
		#valid部分
		x_va_list.append(x_ma[j,:,:] for j in vaild_index)
		y_va_list.append(y_ma[j] for j in vaild_index)
		#train部分
		x_ma_list.append(x_ma[j,:,:] for j in other_index)
		y_ma_list.append(y_ma[j] for j in other_index)

	return x_ma_list,y_ma_list,x_va_list,y_va_list

def predict(x, weight, bias):
	return np.sum(x * w) + b

def scale(x_ma):
	min_x = np.min(x_ma, axis=0)
	max_x = np.max(x_ma, axis=0)
	return (x_ma-min_x)/float(max_x-min_x)

def predict_(x_ma,weight,bias):
	return x_ma*weight+bias

def error_(predict,y_ma):
	return predict-y_ma

def loss_(error):
	return np.sqrt(np.mean(error ** 2))	

def select(test, start, time, features):
	op_list = []
	for f in features:
		op_list += [test[f][start : start + time]]
	return op_list

def train(x_ma,y_ma,x_va,y_va,weight,bias,lr,weight_lr,bias_lr,epoch):
	x_ma = scale(x_ma)
	x_va = scale(x_va)
	#資料數
	row_num = x_ma.shape[0]
	#特徵數=18*9=162
	feature_num = x_ma.shape[1]*9
	assert feature_num == 162,'feature_num error!!!'
	#攤平所有特徵
	x_ma = np.reshape(row_num,(-1,feature_num))
	print('x shape = ',x_ma.shape)

	for i in tqdm(epoch):
		predict = predict(x_ma,weight,bias)
		error = error(predict,y_ma)

		weight_gd = np.mean(error*x_ma)
		weight_lr = weight_lr + weight_gd**2
		weight = weight - lr / np.sqrt(weight_lr) * weight_gd

		bias_gd = np.mean(np.sum(error))
		bias_lr = bias_lr + bias_gd**2
		bias = bias -  lr / np.sqrt(bias_lr) * bias_gd

		if i%100 == 0:
			predict_v = predict(x_va)
			error_v = error(predict_v,y_va)
			loss_t = loss(error)
			loss_v = loss(error_v)
			print('[Epoch {}]: loss: {}'.format(epoch, loss_t)+'\n')
			print('valid loss: {}'.format(loss_v)+'\n')

			with open ('allf_loss_t.txt','a') as f:
				f.write(str(loss_t))
			with open ('allf_loss_v.txt','a') as f:
				f.write(str(loss_v))

	return weight,bias,weight_lr,bias_lr

def main():	
	x_ma,y_ma = pre_traindata('train.csv')
	x_ma_list,y_ma_list,x_va_list,y_va_list = ten_fold(x_ma,y_ma)

	feature_num = x_ma.shape[1]*9

	bias = 0.0
	weight = np.ones((162, 1))

	bias_lr = 0.0
	weight_lr = np.zeros((162, 1))


	for i in range(10):
		x_ma = x_ma_list[i]
		y_ma = y_ma_list[i]
		x_va = x_va_list[i]
		y_va = y_va_list[i]

		weight,bias,weight_lr,bias_lr = train(x_ma,y_ma,x_va,y_va,weight,bias,lr,weight_lr,bias_lr,epoch)
		
	with open('allf_b.npy','wb') as f:
		np.save(f,bias)
	with open('allf_w.npy','wb') as f:
		np.save(f,weight)

	"""
	bias = np.load('pm25_t1_b.npy')
	weight = np.load('pm25_t1_w.npy')
	"""
	test_x_ma = np.load('test_x.npy')
	#test_x_ma = (test_x_ma - min_x) / (max_x - min_x)
	#test_x_ma = np.reshape(test_x_ma, (-1, feature_num))
	pred_y = np.dot(test_x_ma,weight)+bias
	#predict = model.predict_test(X_test)
	with open('output_me.csv', 'w') as f:
		print('id,value', file=f)
		for (i, p) in enumerate(pred_y) :
			print('id_{},{}'.format(i, p[0]), file=f)

w = np.load('my_w.npy')
b = np.load('my_b.npy')

inputfile = sys.argv[1]
outfile = sys.argv[2]

test = pre_testdata(inputfile)
print(len(test)/18)
with open(outfile, "w") as f:
	f.write("id,value\n")
	for d in range(260):
		test_data = np.array(select(test[d*18:(d+1)*18],9-7,7,[7,9,12]))
		predict_op = predict(test_data, w, b)
		f.write("id_"+str(d)+","+str(predict_op)+"\n")


