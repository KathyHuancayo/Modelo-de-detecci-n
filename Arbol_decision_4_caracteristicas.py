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

df = pd.read_csv('capturas_4_2.csv', keep_default_na=False, na_values=[""])
df.head(10)

#LIMPIEZA DE DATOS NULOS
df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)

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

df1 = df[['Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]   

features =  ['Src_IP','Src_Port','Dst_IP','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']

y=df.Output
print(y)


df.shape
df.dtypes

print("CANTIDAD DE DATOS PARA CADA CLASE: ")
print(df.groupby('Output').size())

# SELECCIONAR CARACTERÍSTICAS MÁS RELEVANTES
from sklearn.feature_selection import SelectKBest
X=df.drop(['Output'], axis=1)
y=df['Output']
 
best=SelectKBest(k=5)
X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
print(X.columns[selected])

#-----------------------GRAFICAR EL GRADO DE CORRELACIÓN ENTRE LAS VARIABLES------------
import seaborn as sb
used_features =X.columns[selected] 
colormap = plt.cm.viridis
plt.figure(figsize=(7,7))
#plt.title('Pearson Correlation of Features', y=1.05, size=15)
sb.heatmap(df[used_features].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
df[used_features].corr(method="pearson")


#OBTENIENDO DATOS DE TEST(PRUEBA) Y TRAIN (ENTRENAMIENTO)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df, test_size = 0.1, random_state = 100)
y_train=X_train['Output']
y_test=X_test['Output']


# List of values to try for max_depth:
max_depth_range = list(range(1,30 ))
# List to store the average RMSE for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,min_impurity_decrease=0,max_features=5,random_state=None,min_samples_leaf=5, 
                                  min_weight_fraction_leaf=0.,max_leaf_nodes=None,presort='deprecated',
                                             max_depth = depth,min_impurity_split=None,class_weight='balanced'
                                             ,ccp_alpha=0.0001)  
    clf.fit(X_train[used_features].values,
    y_train)
    score = clf.score(X_test[used_features], y_test)
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

# Correr desde aquí hasta la precisión en el set de test
from sklearn.tree import DecisionTreeClassifier
# INSTANCIAR EL CLASSIFIER
classifier=DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=2,min_impurity_decrease=0,max_features=5,random_state=None,min_samples_leaf=5, 
                                  min_weight_fraction_leaf=0.,max_leaf_nodes=None,presort='deprecated',
                                             max_depth = 10,min_impurity_split=None,class_weight='balanced',
                                             ccp_alpha=0.0001)
# ENTRENAR EL CLASSIFIER
classifier.fit(
    X_train[used_features].values,
    y_train
)
predictions = classifier.predict(X_test[used_features])

print(classifier.score(X_test[used_features],y_test))


# VALIDAR MODELO
name='ARBOL DE DECISIÓN'
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(classifier, X_train[used_features], y_train,scoring='accuracy', cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100, results.std()*100))
print ('CROSS-VALIDATION SCORES:')
print(results)

#------------------------------REPORTE DE RESULTADOS POR CLASE-------------------------------
print(classification_report(y_test,predictions))

print("PRECISIÓN PARA DETECTAR A LA BOTNET (Zeus y Ares) ", metrics.precision_score(y_test, predictions, pos_label=1)*100)

print("RECALL PARA DETECTAR A LA BOTNET(Zeus y Ares): ", metrics.recall_score(y_test, predictions, pos_label=1)*100)

print("PRECISIÓN PARA DETECTAR A UN BENIGNO ", metrics.precision_score(y_test, predictions, pos_label=0)*100)

print("RECALL PARA DETECTAR A LA BENIGNO: ", metrics.recall_score(y_test, predictions, pos_label=0)*100)


print("PRECISIÓN PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET ", metrics.precision_score(y_test, predictions,average=None)*100)

print("RECALL PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET: ", metrics.recall_score(y_test, predictions, average=None)*100)

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)

print("MATRIZ DE CONFUSIÓN PARA VALIDACION: ",multilabel_confusion_matrix(y_test, predictions))


#------------1.GRÁFICO DEL ÁRBOL DE DECISIÓN--------------------------------
#DIBUJAR EL ÁRBOL
from sklearn import tree
from io import StringIO
from IPython.display import Image
#PINTAR EL ÁRBOL
out = StringIO()
tree.export_graphviz(classifier, out_file='Tree_CR.dot') 

 
# PREDECIR BOTNET BLACKOUT:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (122,38,122,19,61,0)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET BLUE:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (118,34,118,17,59,1)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET LIPHYRA:

X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (121,37,121,18,60,2)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET BLACK ENERGY:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (116,32,116,16,58,3)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET ZEUS:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (118,34,118,17,59,4)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET ZYKLON:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (120,36,120,18,60,5)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET CITADEL:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (121,37,121,18,60,6)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET CITADEL2:
X_test = pd.DataFrame(columns=('TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Output'))

X_test.loc[0] = (38,38,38,28,9,7)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


