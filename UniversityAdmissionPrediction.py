# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:31:51 2020

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Admission_Predict_Ver1.1.csv')
df.drop(labels='Serial No.',inplace=True,axis=1) #Dropped S.No column because it has no purpose for us
df.apply(lambda x: sum(x.isnull()),axis=0)
df.apply(lambda x: sum(x==0),axis=0)
x=df.iloc[:,:7].values
y=df.iloc[:,7:8].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train=(y_train>0.5)
y_test=(y_test>0.5)
from sklearn.linear_model.logistic import LogisticRegression
cls1 =LogisticRegression(random_state =0)
lr=cls1.fit(x_train, y_train)
y_pred =lr.predict(x_test)
y_pred

import pickle
pickle.dump(lr,open('UniversityAdmissionPrediction.pkl','wb'))
model=pickle.load(open('UniversityAdmissionPrediction.pkl','rb'))
