import numpy as np
import pandas as pd
import csv
import sys
import math
#import random

from sklearn.linear_model import LogisticRegression

"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDRegressor,LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
#from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest ,chi2,f_classif
#from sklearn.feature_selection import f_regression
#from sklearn.tree import DecisionTreeRegressor  
#from sklearn.ensemble import RandomForestRegressor  
#from sklearn.metrics.regression import mean_squared_error, r2_score  
from sklearn.metrics import accuracy_score
"""
import pickle as pkl

####################
#全域變數
####################
NUM_FEATURE = 120
OTHER_FEATURE = 1

ITERATION = 5000
ETA = 1e-2
ADAGRAD = 0

NORMALIZE = 1
REGULARIZE = 0
LAMBDA = 0.01


#輸出精準度
np.set_printoptions(precision = 5, suppress = True)
#避免underflow
np.seterr(divide='ignore', invalid='ignore')


#######################
#讀資料
#######################
train_x = []
with open(sys.argv[3], "r", encoding="big5") as f:
	for line in list(csv.reader(f))[1:]:
		train_x.append([float(x) for x in line])
train_x = np.array(train_x)

train_y = []
with open(sys.argv[4], "r", encoding="big5") as f:
	for line in list(csv.reader(f)):
		train_y.append([int(x) for x in line])
train_y = np.array(train_y)


test_x = []
with open(sys.argv[5], "r", encoding="big5") as f:
	for line in list(csv.reader(f))[1:]:
		test_x.append([float(x) for x in line])
test_x = np.array(test_x)


##########################
#增加特徵
##########################
for i in range(train_x.shape[1]):
		if max(train_x[:,[i]]) > 1:
			x1 = train_x[:, [i]] ** 3
			x2 = train_x[:,[i]] ** 0.5
			#x3 = train_x[:,[i]] ** 2
			train_x = np.concatenate((train_x, x1,x2), axis = 1)
#NUM_FEATURE += 3
#train_x = np.concatenate((train_x, x1, x2, x3), axis = 1)

for i in range(test_x.shape[1]):
		if max(test_x[:,[i]]) > 1:
			x1 = test_x[:, [i]] ** 3
			x2 = test_x[:, [i]] ** 0.5
			#x3 = test_x[:, [i]] ** 2
			test_x = np.concatenate((test_x, x1,x2), axis = 1)
#test_x = np.concatenate((test_x, x1, x2, x3), axis = 1)

#print(train_x.shape)
#print(test_x.shape)
#print(train_y.shape)
#########################
#scaling
#########################
"""
for i in range(train_x.shape[1]):
	max_ = np.max(train_x[:,i])
	if max_ > 1:
		#min_ = np.min(train_x[:,i])
		mean = np.mean(train_x[:,i])
		std = np.std(train_x[:,i])
		#train_x[:,i] = (train_x[:,i]-min_)/(max_-min_)
		train_x[:,i] = (train_x[:,i]-mean)/std
"""

mean = np.mean(train_x,axis=0)
std = np.std(train_x,axis=0)
train_x = (train_x-mean)/std

test_x = (test_x-mean)/std

#########################
#挑選特徵
#########################
"""
sel = SelectKBest(f_classif, k=NUM_FEATURE)
train_x = sel.fit_transform(train_x, train_y)
print(train_x.shape)

#記住選擇的特徵索引
index = sel.get_support()
need_fea = []
for i in range(len(index)):
    if index[i] == True:
        need_fea.append(i)
print(need_fea)

with open ('need_fea.pkl','wb') as f:
	pkl.dump(need_fea,f)
"""
need_fea = pkl.load(open('need_fea.pkl','rb'))
new_test = []
for i in range(test_x.shape[0]):
	new_test.append([])
	for j in range(NUM_FEATURE):
		new_test[i].append(test_x[i][int(need_fea[j])])
new_test = np.array(new_test)
#print(new_test.shape)
###########################

#classifer = LogisticRegression(C=10000,random_state=42,max_iter=ITERATION,n_jobs=-1,class_weight='balanced'
#	,solver='newton-cg')
#print('10000')
"""
classifer = LogisticRegression()
#classifer.fit(train_x,train_y)
scores = cross_val_score(classifer, train_x, train_y, cv=StratifiedKFold(10).split(train_x,train_y), scoring='accuracy')
print(scores)
classifer.fit(train_x,train_y)
y_pred = classifer.predict(train_x)
print ('accuracy train:%.3f'%(accuracy_score(train_y,y_pred)))  

with open (sys.argv[6]+'_model.pkl','wb') as f:
	pkl.dump(classifer,f)
"""
classifer = pkl.load(open('best_model.pkl','rb'))
result = classifer.predict(new_test)
with open(sys.argv[6], "w") as f:
	f.write("id,label\n")
	for i in range(len(result)):
		#f.write(repr(i + 1) + "," + repr(result[i]) + "\n")
		f.write('%d,%d\n' %(i+1, result[i]))


