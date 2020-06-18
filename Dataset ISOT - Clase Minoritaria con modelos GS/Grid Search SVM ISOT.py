# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:50:35 2020

@author: Sony
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn import metrics
from sklearn.svm import LinearSVC

datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data//Clase_Minoritaria.csv')

df=pd.DataFrame(datos)

X = datos[['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state = 0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

from sklearn.model_selection import GridSearchCV
model=sklearn.svm.LinearSVC(penalty='l2',dual=False, max_iter=12000,class_weight='balanced')
tol=[1e-20,1e-10,1e-5]
C=[10,100,175]
parameters = dict(tol=tol,C=C)
clf = GridSearchCV(model, parameters, cv = 5)
clf.fit(X_train, y_train)

clf.best_score_

clf.best_estimator_

clf.best_params_

