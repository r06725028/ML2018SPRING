import sys
import csv 
import math
import random
import numpy as np

"""
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import SGDRegressor,LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest ,chi2
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor   
from sklearn.metrics.regression import mean_squared_error, r2_score  
"""
from sklearn.ensemble import RandomForestRegressor 
#from sklearn.feature_selection import SelectKBest ,chi2
#from sklearn.feature_selection import f_regression


import pickle as pkl

"""
fea_num = 80
###############
data = []    
#一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列為header沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
                #if float(r[i]) < 0:data[(n_row-1)%18].append(float(0))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 總共有18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

#x = np.concatenate((x,x**2), axis=1)
# 增加平方項

#x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
# 增加bias項       



#############################
#X_train, X_test, y_train, y_test = train_test_split(x,y)
##############################
#X_scaler = StandardScaler()
#X_train = X_scaler.fit_transform(X_train)

#y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

#print(x.shape)
#print(y.shape)

########################

sel = VarianceThreshold(threshold=.08)
#选择方差大于0.8的特征
X_train = sel.fit_transform(X_train)
print(X_train.shape)

sel = SelectKBest(f_regression, k=fea_num)
x = sel.fit_transform(x, y)
print(x.shape)

index = sel.get_support()
need_fea = []
for i in range(len(index)):
    if index[i] == True:
        need_fea.append(i)
print(need_fea)

with open('need_fea.pkl','wb') as f:
    pkl.dump(need_fea,f)

new_test = []
#need_fea = [77, 78, 79, 80, 84, 85, 88, 89, 240, 241, 242, 251]
#[83, 84, 85, 87, 88, 89]
for i in range(x.shape[0]):
    new_test.append([])
    for j in range(fea_num):
        new_test[i].append(x[i][int(need_fea[j])])
new_test = np.array(new_test)
print(new_test.shape)

#######################################
#forest = RandomForestRegressor(criterion='mse',random_state=42,n_jobs=-1) 

grid = {'min_samples_leaf':list(range(3, 18)),
        'max_depth': list(range(3, 10)),
        'n_estimators': list(range(1,30))}

search = GridSearchCV(estimator=RandomForestRegressor(criterion='mse',random_state=42,n_jobs=15), 
                      param_grid=grid, cv=5, n_jobs=15)

search.fit(x,y)

print (search.best_score_)
print (search.best_params_)

forest = search.best_estimator_ 
#forest.fit(X_train,y_train)  
y_pred = forest.predict(x)  
#y_test_pred = forest.predict(X_test)  
print ('MSE6 train:%.3f'%(mean_squared_error(y,y_pred)))  
print ('R^2 train:%.3f'%(r2_score(y,y_pred)))  



#np.save('reg00_model.npy',regressor)
with open('fors100_model.pkl','wb') as f:
    pkl.dump(forest,f)
"""
#regressor = np.load('reg00_model.npy','rb')
with open('best_model.pkl','rb') as f:
    forest = pkl.load(f)

with open('need_fea.pkl','rb') as f:
    need_fea = pkl.load(f)
"""
y_pred = forest.predict(x)  
#y_test_pred = forest.predict(X_test)  
print ('MSE6 train:%.3f'%(mean_squared_error(y,y_pred)))  
print ('R^2 train:%.3f'%(r2_score(y,y_pred))) 
""" 
######################

test_x = []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[n_row//18].append(float(r[i]) )
    else :
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
                #if float(r[i]) < 0:test_x[n_row//18].append(0)
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

#test_x = np.concatenate((test_x,test_x**2), axis=1)
# 增加平方項

#test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
# 增加bias項  

new_test = []
#need_fea = [77, 78, 79, 80, 84, 85, 88, 89, 240, 241, 242, 251]
#[83, 84, 85, 87, 88, 89]
for i in range(test_x.shape[0]):
    new_test.append([])
    for j in range(len(need_fea)):
        new_test[i].append(test_x[i][int(need_fea[j])])
new_test = np.array(new_test)
#print(new_test.shape)


###########################
#test_scaler = StandardScaler()
#new_test = test_scaler.fit_transform(new_test)

predict = forest.predict(new_test)

with open(sys.argv[2], 'w') as f:
        print('id,value', file=f)
        for (i, p) in enumerate(predict) :
            print('id_{},{}'.format(i, p), file=f)

#print(forest.get_params())

