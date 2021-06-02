# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:36:25 2021

@author: Kulsum
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("50_Startups.csv")
X=data.iloc[:,:4]
y=data.iloc[:,4:]
states=pd.get_dummies(X['State'],drop_first=True)
X=X.drop('State',axis=1)
X=pd.concat([X,states],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)