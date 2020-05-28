# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:46:20 2019

@author: user
"""
#PROCESAMIENTO
import time
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
#CARGAR LOS DATOS
os.getcwd()

os.chdir("C://Users//Sony//Desktop//TESIS 2")

df = pd.read_csv('CIC_AWS_Filtrado.csv')
df.head(10)

#PREPARAR LOS DATOS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import sklearn.metrics
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

df1 = df[['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]   

features =  ['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']

y=df.Output
print(y)


df1.shape
df.dtypes
print("CANTIDAD DE DATOS PARA CADA CLASE: ")
print(df.groupby('Output').size())

#OBTENIENDO DATOS DE TEST(PRUEBA) Y TRAIN (ENTRENAMIENTO)
X_train, X_test, y_train, y_test = train_test_split( df1, y, test_size = 0.1,
                                                    random_state =0 )

X_train.shape
X_test.shape
y_train.shape
y_test.shape


# List of values to try for max_depth:
max_depth_range = list(range(10,80))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,max_features=19,min_samples_leaf=1, max_depth = depth)  
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test) 
    accuracy.append(score)
    
df = pd.DataFrame({"Max Depth": max_depth_range, "Average Accuracy": accuracy})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
#GRAFICAR LOS RESULTADOS
plt.plot(max_depth_range, accuracy, color='b', label='Exactitud')
plt.legend()
plt.ylabel('exactitud')
plt.xlabel('Nivel de profundidad')
plt.show()


#CREAR EL MODELO
start = time.time()
classifier=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=20, max_features=19, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=10,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='best') 
classifier=classifier.fit(X_train,y_train)
end = time.time()
print ("Decision Tree", end - start)
predictions = classifier.predict(X_test)

# VALIDAR MODELO
name='ARBOL DE DECISIÓN'
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(classifier, X_train, y_train,scoring='accuracy', cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100))
print ('CROSS-VALIDATION SCORES:')
print(results)

print(classifier.score(X_test,y_test))


print(classification_report(y_test,predictions))
 
#------------------------------REPORTE DE RESULTADOS POR CLASE-------------------------------

print("PRECISIÓN PARA DETECTAR A UN BENIGNO ", metrics.precision_score(y_test, predictions, pos_label=0)*100)

print("PRECISIÓN PARA DETECTAR A LA BOTNET ", metrics.precision_score(y_test, predictions, pos_label=1)*100)

print("RECALL PARA DETECTAR A LA BENIGNO: ", metrics.recall_score(y_test, predictions, pos_label=0)*100)

print("RECALL PARA DETECTAR A LA BOTNET: ", metrics.recall_score(y_test, predictions, pos_label=1)*100)


print("MATRIZ DE CONFUSIÓN PARA VALIDACION: ",multilabel_confusion_matrix(y_test, predictions))

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)




# PREDECIR BENIGNO

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (3389,17,119996043,40,56,1175,45187,0,0,29.375,806.9107143,386.3627403,0.800026381,0.333344325,0.466682056,1175,45187,56,40,0)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOT

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))
X_test.loc[0] = (8080,6,10869,3,4,326,129,0,0,108.6666667,32.25,41862.17683,644.0334897,276.0143527,368.019137,326,129,4,3,1)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))




#------------1.GRÁFICO DEL ÁRBOL DE DECISIÓN--------------------------------
#DIBUJAR EL ÁRBOL
from sklearn import tree
from io import StringIO
from IPython.display import Image
#PINTAR EL ÁRBOL
out = StringIO()
tree.export_graphviz(classifier, out_file='CIC-AWS-MODELO.dot')

 





