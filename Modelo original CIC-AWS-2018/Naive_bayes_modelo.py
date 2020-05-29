# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:46:20 2019

@author: user
"""

import os 
import time
#PROCESAMIENTO
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


#CARGAR LOS DATOS

datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//CIC-AWS-2018.csv')

df=pd.DataFrame(datos)

X = datos[['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state = 0)

start = time.time()
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
model=  GaussianNB(priors=None, var_smoothing=1e-10)
model.fit(X_train,y_train)
end = time.time()
print ("Naive Bayes", end - start)
# Predicting the Test set results
y_pred = model.predict(X_test)


print(model.score(X_test,y_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold

# VALIDAR MODELO
name='NAIVE BAYES'
results = cross_val_score(model, X_train, y_train,scoring='accuracy', cv=5)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100))
print ('CROSS-VALIDATION SCORES:')
print(results)
#------------------------REPORTE DE RESULTADOS POR CLASE----------------------------------
from sklearn.metrics import classification_report
from sklearn import metrics
import sklearn.metrics

print(classification_report(y_test,y_pred))


print("PRECISIÓN PARA DETECTAR A UN BENIGNO ", metrics.precision_score(y_test, y_pred, pos_label=0)*100)

print("PRECISIÓN PARA DETECTAR A LA BOTNET ", metrics.precision_score(y_test, y_pred, pos_label=1)*100)

print("RECALL PARA DETECTAR A LA BENIGNO: ", metrics.recall_score(y_test, y_pred, pos_label=0)*100)

print("RECALL PARA DETECTAR A LA BOTNET: ", metrics.recall_score(y_test, y_pred, pos_label=1)*100)

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)*100)



#-----------------------RESULTADOS DEL MODELO------------------------------------

print("MATRIZ DE CONFUSIÓN: ")
print(cm)
from sklearn import metrics
import sklearn.metrics
from sklearn.metrics import accuracy_score
print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, y_pred,normalize=True)*100)


# PREDECIR BENIGNO

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (3389,17,119996043,40,56,1175,45187,0,0,29.375,806.9107143,386.3627403,0.800026381,0.333344325,0.466682056,1175,45187,56,40,0)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOT

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (8080,6,10869,3,4,326,129,0,0,108.6666667,32.25,41862.17683,644.0334897,276.0143527,368.019137,326,129,4,3,1)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

