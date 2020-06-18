# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:20:00 2020

@author: Sony
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
 
%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn import metrics

#CARGAR LOS DATOS
datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data//Clase_Mayoritaria.csv')

df=pd.DataFrame(datos)

X = datos[['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape


k_range = list(range(1,20))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k,algorithm='kd_tree', 
                     leaf_size=30, 
                     metric='minkowski',
                     metric_params=None, 
                     n_jobs=1, 
                     p=2,
                     weights='uniform')
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
df = pd.DataFrame({"Max K": k_range, "Average Accuracy":scores })
df = df[["Max K", "Average Accuracy"]]
print(df.to_string(index=False))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])

start = time.time()
knn = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')
knn.fit(X_train, y_train)
end = time.time()
print ("K-Nearest Neighbors", end - start)
predictions=knn.predict(X_test)

# VALIDAR MODELO
results = cross_val_score(knn, X_train, y_train,scoring='accuracy', cv=5)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100))
print ('CROSS-VALIDATION SCORES:')
print(results)
 
#------------------------------REPORTE DE RESULTADOS POR CLASE-------------------------------
print(classification_report(y_test,predictions))

print("PRECISIÓN PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET ", metrics.precision_score(y_test, predictions,average=None)*100)

print("RECALL PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET: ", metrics.recall_score(y_test, predictions, average=None)*100)

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)

print(knn.score(X_test,y_test))

#------------------------------MATRIZ DE CONFUSIÓN PARA VALIDACIÓN-------------------------------
x_axis_labels = ['Zeus','Citadel','Citadel2'] # labels for x-axis
y_axis_labels =  ['Zeus','Citadel','Citadel2'] # labels for y-axis

print("MATRIZ DE CONFUSIÓN PARA VALIDACION: ")
cm1=confusion_matrix(y_test, predictions)
print(cm1)
plt.figure(figsize=(12, 12))
sns.heatmap(cm1,xticklabels=x_axis_labels, yticklabels=y_axis_labels ,annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# PREDECIR BOTNET ZEUS:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (57366,53,17,216,1,1,34,118,34,34,34,118,703703.7037,9259.259259,4629.62963,4629.62963,17,59,0,0,4)
y_pred = knn.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = knn.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET CITADEL:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (63587,53,17,370,1,1,37,121,37,37,37,121,427027.027,5405.405405,2702.702703,2702.702703,18,60,0,0,6)
y_pred = knn.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = knn.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET CITADEL2:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (55399,53,17,85,1,1,37,83,37,37,37,83,1411764.706,23529.41176,11764.70588,11764.70588,18,41,0,0,7)
y_pred = knn.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = knn.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

