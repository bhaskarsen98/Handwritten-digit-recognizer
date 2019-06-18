#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:45:14 2019

@author: bhaskar
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np
from numpy.random import permutation
import math
from sklearn import preprocessing
#%%read dataset
df=pd.read_csv('train.csv')
df_=pd.read_csv('test.csv')

#%%
"""
#%%validation
    #test train split
random_indices=permutation(df.index)
test_cutoff=math.floor(len(df)*.33)
test=df.loc[random_indices[0:test_cutoff]]
train=df.loc[random_indices[test_cutoff:]]
    #defining features
y_col=['label']
x_col=list(df.columns)
x_col=x_col[1:]
    #defining model
Knn=KNeighborsClassifier(n_neighbors=5)
Knn.fit(train[x_col],train[y_col])
    #prediction
predicted=Knn.predict(test[x_col])
    #accuracy
actual_label=test['label']
actual_label=np.array(actual_label)
true_positives=0
for index,pred in enumerate(predicted):
    if(pred==actual_label[index]):
        true_positives+=1
acc=(true_positives/len(predicted))*100
print('accuracy=',acc,'%')
    #validation_accuracy=97.16666666666667 %
#%%
"""
#%%defining features and classes
y_col=['label']
x_col=list(df.columns)
x_col=x_col[1:]
#%% define model
test=df_
train=df
Knn=KNeighborsClassifier(n_neighbors=5)
    #train
Knn.fit(train[x_col],train[y_col])
    #predict classes
predicted=Knn.predict(test[x_col])
predicted_=pd.DataFrame(predicted)
predicted_.columns=['Label']
predicted_.to_csv('prediction.csv',index=True) 

