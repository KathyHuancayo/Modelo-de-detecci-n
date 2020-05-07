# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:46:20 2019

@author: user
"""
#PROCESAMIENTO
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

os.chdir("C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data")

df = pd.read_csv('capturas_4.csv', keep_default_na=False, na_values=[""])
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

df1 = df[['Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration',
          'Tot_Fwd_Pkts','Tot_Bwd_Pkts','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts',
          'Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd_Pkt_Len_Mean','Fwd Pkt Len Std',
          'Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd_Pkt_Len_Mean','Bwd Pkt Len Std',
          'Flow_Byts/s','Flow_Pkts/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max',
          'Flow IAT Min','Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max',
          'Fwd IAT Min','Bwd IAT Tot','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max',
          'Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
          'Fwd Header Len','Bwd Header Len','Fwd_Pkts/s','Bwd_Pkts/s','Pkt Len Min',
          'Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt',
          'SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt',
          'CWE Flag Count','ECE Flag Cnt','Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg',
          'Bwd Seg Size Avg','Fwd Byts/b Avg','Fwd Pkts/b Avg','Fwd Blk Rate Avg',
          'Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Subflow_Fwd_Pkts',
          'Subflow_Fwd_Byts','Subflow_Bwd_Pkts','Subflow_Bwd_Byts','Init Fwd Win Byts',
          'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean'
          ,'Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max',
          'Idle Min']]   

features =  ['Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration',
          'Tot_Fwd_Pkts','Tot_Bwd_Pkts','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts',
          'Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd_Pkt_Len_Mean','Fwd Pkt Len Std',
          'Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd_Pkt_Len_Mean','Bwd Pkt Len Std',
          'Flow_Byts/s','Flow_Pkts/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max',
          'Flow IAT Min','Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max',
          'Fwd IAT Min','Bwd IAT Tot','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max',
          'Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
          'Fwd Header Len','Bwd Header Len','Fwd_Pkts/s','Bwd_Pkts/s','Pkt Len Min',
          'Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt',
          'SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt',
          'CWE Flag Count','ECE Flag Cnt','Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg',
          'Bwd Seg Size Avg','Fwd Byts/b Avg','Fwd Pkts/b Avg','Fwd Blk Rate Avg',
          'Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Subflow_Fwd_Pkts',
          'Subflow_Fwd_Byts','Subflow_Bwd_Pkts','Subflow_Bwd_Byts','Init Fwd Win Byts',
          'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean'
          ,'Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min']

y=df.Output
print(y)


df.shape
df.dtypes


#OBTENIENDO DATOS DE TEST(PRUEBA) Y TRAIN (ENTRENAMIENTO)
X_train, X_test, y_train, y_test = train_test_split( df1, y, test_size = 0.1,
                                                    random_state =100 )

X_train.shape
X_test.shape
y_train.shape
y_test.shape

# List of values to try for max_depth:
max_depth_range = list(range(1,30 ))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,min_impurity_decrease=0,max_features=81,random_state=None,min_samples_leaf=2, 
                                  min_weight_fraction_leaf=0.,max_leaf_nodes=None,presort='deprecated',
                                             max_depth = depth,min_impurity_split=None)  
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)
    
df = pd.DataFrame({"Max Depth": max_depth_range, "Average Accuracy": accuracy})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))

#CREAR EL MODELO

classifier=DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,min_impurity_decrease=0,max_features=81,random_state=None,min_samples_leaf=2, 
                                  min_weight_fraction_leaf=0.,max_leaf_nodes=None,presort='deprecated',
                                             max_depth = 10,min_impurity_split=None
                                             ,ccp_alpha=0.0001) 
classifier=classifier.fit(X_train,y_train)
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

print("PRECISIÓN PARA DETECTAR A LA BOTNET ", metrics.precision_score(y_test, predictions,average=None)*100)

print("RECALL PARA DETECTAR A LA BOTNET: ", metrics.recall_score(y_test, predictions, average=None)*100)

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)

print("MATRIZ DE CONFUSIÓN PARA VALIDACION: ",multilabel_confusion_matrix(y_test, predictions))




#------------1.GRÁFICO DEL ÁRBOL DE DECISIÓN--------------------------------
#DIBUJAR EL ÁRBOL
from sklearn import tree
from io import StringIO
from IPython.display import Image
#PINTAR EL ÁRBOL
out = StringIO()
tree.export_graphviz(classifier, out_file='treeMacarena.dot')

 
# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration',
          'Tot_Fwd_Pkts','Tot_Bwd_Pkts','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts',
          'Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd_Pkt_Len_Mean','Fwd Pkt Len Std',
          'Bwd Pkt Len Max','Bwd Pkt Len Min','Bwd_Pkt_Len_Mean','Bwd Pkt Len Std',
          'Flow_Byts/s','Flow_Pkts/s','Flow IAT Mean','Flow IAT Std','Flow IAT Max',
          'Flow IAT Min','Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max',
          'Fwd IAT Min','Bwd IAT Tot','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max',
          'Bwd IAT Min','Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
          'Fwd Header Len','Bwd Header Len','Fwd_Pkts/s','Bwd_Pkts/s','Pkt Len Min',
          'Pkt Len Max','Pkt Len Mean','Pkt Len Std','Pkt Len Var','FIN Flag Cnt',
          'SYN Flag Cnt','RST Flag Cnt','PSH Flag Cnt','ACK Flag Cnt','URG Flag Cnt',
          'CWE Flag Count','ECE Flag Cnt','Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg',
          'Bwd Seg Size Avg','Fwd Byts/b Avg','Fwd Pkts/b Avg','Fwd Blk Rate Avg',
          'Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Subflow_Fwd_Pkts',
          'Subflow_Fwd_Byts','Subflow_Bwd_Pkts','Subflow_Bwd_Byts','Init Fwd Win Byts',
          'Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean'
          ,'Active Std','Active Max','Active Min','Idle Mean','Idle Std','Idle Max',
          'Idle Min','Output'))

X_test.loc[0] = (192_168_50_15,62423,192_168_50_88,53,17,208,1,1,34,118,34,34,34,0,118,118,118,0,730769.2308,9615.384615,208,0,208,208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,8,4807.692308,4807.692308,34,118,62,48.49742261,2352,0,0,0,0,0,0,0,0,1,93,34,118,0,0,0,0,0,0,0,17,0,59,0,0,0,8,0,0,0,0,0,0,0,0,1)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (192_168_50_15,58093,192_168_50_88,53,17,218,1,1,34,118,34,118,697247.7064,9174.311927,4587.155963,4587.155963,17,59,0,0,1)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BLACKOUT

X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_16,62238,192_168_50_88,53,17,246,1,1,37,121,37,121,642276.4228,8130.081301,4065.04065,4065.04065,18,60,2)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_32,53240,192_168_50_88,53,17,224,1,1,32,116,32,116,660714.2857,8928.571429,4464.285714,4464.285714,16,58,3)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_34,50145,192_168_50_88,53,17,211,1,1,34,118,34,118,720379.1469,9478.672986,4739.336493,4739.336493,17,59,4)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))



# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_14,52118,192_168_50_88,53,17,199,1,1,36,120,36,120,783919.598,10050.25126,5025.125628,5025.125628,18,60,5)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BLUE
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_30,56062,192_168_50_88,53,17,225,1,1,37,121,37,121,702222.2222,8888.888889,4444.444444,4444.444444,18,60,6)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR citadel2
X_test = pd.DataFrame(columns=('Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (192_168_50_31,57967,192_168_50_88,53,17,10000498,3,1,114,38,38,38,15.19924308,0.399980081,0.299985061,0.09999502,28,9,7)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))







