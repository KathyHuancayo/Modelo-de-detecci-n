# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:13:51 2020

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

datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//CIC_AWS_Filtrado.csv')

df=pd.DataFrame(datos)

X = datos[['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

start = time.time()
model=sklearn.svm.LinearSVC(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=200000,
          multi_class='ovr', penalty='l2', random_state=None, tol=1e-10,
          verbose=0)
model.fit(X_train,y_train)
end = time.time()
print ("SVC Linear", end - start)
predictions=model.predict(X_test)

print(classification_report(y_test,predictions))

print("PRECISIÓN PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET ", metrics.precision_score(y_test, predictions,average=None, zero_division='warn')*100)
print("RECALL PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET: ", metrics.recall_score(y_test, predictions, average=None, zero_division='warn')*100)
print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)

# VALIDAR MODELO
results = cross_val_score(model, X_train, y_train,scoring='accuracy', cv=5)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100))
print ('CROSS-VALIDATION SCORES:')
print(results)

# PREDECIR BENIGNO

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (3389,17,119996043,40,56,1175,45187,0,0,29.375,806.9107143,386.3627403,0.800026381,0.333344325,0.466682056,1175,45187,56,40,0)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))

# PREDECIR BOT

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (8080,6,10869,3,4,326,129,0,0,108.6666667,32.25,41862.17683,644.0334897,276.0143527,368.019137,326,129,4,3,1)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))

