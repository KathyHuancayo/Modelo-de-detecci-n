# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:59:33 2020

@author: Sony
"""



import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn import metrics
from sklearn.svm import LinearSVC

datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//CIC_AWS_Filtrado.csv')

df=pd.DataFrame(datos)

X = datos[['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']


from sklearn.model_selection import GridSearchCV
model = KNeighborsClassifier(metric='minkowski',leaf_size=30, algorithm ='kd_tree')
n_neighbors =[4,10,2]
parameters = dict(n_neighbors=n_neighbors)
clf = GridSearchCV(model, parameters, cv = 2)

clf.fit(X, y)

clf.best_estimator_

clf.best_score_

clf.best_params_