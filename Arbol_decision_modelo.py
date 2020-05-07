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

os.chdir("C://Users//Sony//Downloads//PRUEBA DE CONCEPTO//")

df = pd.read_csv('Hoja.csv', keep_default_na=False, na_values=[""])

#EXPLORATORIO

df['Fwd_Pkt_Len_Mean'].hist()
plt.title("Histograma de Fwd_Pkt_Len_Mean" )
plt.xlabel("Fwd_Pkt_Len_Mean")
plt.ylabel("Frecuencia")
plt.show()

df['Bwd_Pkt_Len_Mean'].hist()
plt.title("Histograma de Bwd_Pkt_Len_Mean" )
plt.xlabel("Bwd_Pkt_Len_Mean")
plt.ylabel("Frecuencia")
plt.show()

df['Fwd_Pkts/s'].hist(bins=8)
plt.title("Histograma de Fwd_Pkts/s" )
plt.xlabel("Fwd_Pkts/s")
plt.ylabel("Frecuencia")
plt.show()

df['Bwd_Pkts/s'].hist()
plt.title("Histograma de Bwd_Pkts/s" )
plt.xlabel("Bwd_Pkts/s")
plt.ylabel("Frecuencia")
plt.show()

print(df.head(10))
print(df.shape)
print(df.columns)
print(df.info())
df.describe().to_csv("temp.csv")

print("CANTIDAD DE DATOS PARA CADA CLASE: ")
print(df.groupby('Output').size())

#LIMPIAR LOS VALORES PARA LOS MISSING EVALUES
df.isnull().sum()
sum(df.isnull().values.ravel()) 

#PREPARAR LOS DATOS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

df1 = df[['Dst_Port','Protocol','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
          'Bwd_Pkt_Len_Mean','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts']]   

features =  ['Dst_Port','Protocol','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
             'Bwd_Pkt_Len_Mean','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts']

y=df.Output

df1.shape

#OBTENIENDO DATOS DE TEST(PRUEBA) Y TRAIN (ENTRENAMIENTO)
X_train, X_test, y_train, y_test = train_test_split( df1, y, test_size = 0.2,
                                                    random_state =100 )

X_train.shape
X_test.shape
y_train.shape
y_test.shape

#CREAR EL MODELO
classifier=DecisionTreeClassifier(criterion='gini',splitter='best',min_samples_split=20,min_impurity_decrease=0,max_features=11,random_state=None,min_samples_leaf=5, 
                                  min_weight_fraction_leaf=0.,max_leaf_nodes=None,presort='deprecated',
                                             max_depth = 20,min_impurity_split=None,
                                             class_weight={1:2.664},ccp_alpha=0.0001)
classifier=classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)

# VALIDAR MODELO

scores = cross_val_score( classifier, X_train ,y_train,scoring='precision', cv=10) 
 
#------------------------------REPORTE DE RESULTADOS POR CLASE-------------------------------

print("PRECISIÓN PARA DETECTAR A LA BOTNET ", metrics.precision_score(y_test, predictions, pos_label=1)*100)

print("RECALL PARA DETECTAR A LA BOTNET: ", metrics.recall_score(y_test, predictions, pos_label=1)*100)

print("PRECISIÓN PARA DETECTAR A UN BENIGNO ", metrics.precision_score(y_test, predictions, pos_label=0)*100)

print("RECALL PARA DETECTAR A LA BENIGNO: ", metrics.recall_score(y_test, predictions, pos_label=0)*100)


#------------------------------REPORTE DE RESULTADOS DEL MODELO-------------------------------
name='ARBOL DE DECISIÓN'


print("MATRIZ DE CONFUSIÓN PARA VALIDACION: ",confusion_matrix(y_test,predictions))

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, predictions, normalize=True)*100)

print('PRECISIÓN DEL MODELO: ', sklearn.metrics.precision_score(y_test,predictions)*100)
print('RECALL DEL MODELO: ' , sklearn.metrics.recall_score(y_test,predictions)*100)



name='ARBOL DE DECISIÓN'
print ('CROSS-VALIDATION SCORES:')
print(scores)
msg="%s: %f (%f)" % (name,scores.mean(),scores.std())
print(msg)
print(classifier.score(X_test,y_test))


#------------1.GRÁFICO DEL ÁRBOL DE DECISIÓN--------------------------------
#DIBUJAR EL ÁRBOL
from sklearn import tree
from io import StringIO
from IPython.display import Image
#PINTAR EL ÁRBOL
out = StringIO()
tree.export_graphviz(classifier, out_file='treeMacarena.dot')

#-----------2.GRÁFICO DE AJUSTE DEL ÁRBOL DE DECISIÓN-----------------------------
classifier.tree_.max_depth
train_prec =  []
eval_prec = []
max_deep_list = list(range(3, 24))

for deep in max_deep_list:
    arbol3 = DecisionTreeClassifier(criterion='gini', max_depth=deep)
    arbol3.fit(X_train, y_train)
    train_prec.append(arbol3.score(X_train, y_train))
    eval_prec.append(arbol3.score(X_test, y_test))
#GRAFICAR LOS RESULTADOS
plt.plot(max_deep_list, train_prec, color='r', label='entrenamiento')
plt.plot(max_deep_list, eval_prec, color='b', label='evaluacion')
plt.title('Grafico de ajuste arbol de decision')
plt.legend()
plt.ylabel('exactitud')
plt.xlabel('cant de nodos')
plt.show()

#------------------------------3. GRAFICAR CURVA DE VALIDACIÓN-------------------------
from sklearn.model_selection import validation_curve

train_prec, eval_prec = validation_curve(estimator=classifier, X=X_train,
                                        y=y_train, param_name='max_depth',
                                        param_range=max_deep_list, cv=5)

train_mean = np.mean(train_prec, axis=1)
train_std = np.std(train_prec, axis=1)
test_mean = np.mean(eval_prec, axis=1)
test_std = np.std(eval_prec, axis=1)
# GRAFICAR LAS CURVAS
plt.plot(max_deep_list, train_mean, color='r', marker='o', markersize=5,
         label='entrenamiento')
plt.fill_between(max_deep_list, train_mean + train_std, 
                 train_mean - train_std, alpha=0.15, color='r')
plt.plot(max_deep_list, test_mean, color='b', linestyle='--', 
         marker='s', markersize=5, label='evaluacion')
plt.fill_between(max_deep_list, test_mean + test_std, 
                 test_mean - test_std, alpha=0.15, color='b')
plt.grid()
plt.title('Curva de Validación')
plt.legend(loc='center right')
plt.xlabel('Cant de nodos')
plt.ylabel('exactitud')
plt.show() 

#----------------------------4. GRAFICAR CURVA DE APRENDIZAJE-------------------------
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                        X=X_train, y=y_train, 
                        train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
                        n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# GRAFICAR LAS CURVAS
plt.plot(train_sizes, train_mean, color='r', marker='o', markersize=5,
         label='entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, 
                 train_mean - train_std, alpha=0.15, color='r')
plt.plot(train_sizes, test_mean, color='b', linestyle='--', 
         marker='s', markersize=5, label='evaluacion')
plt.fill_between(train_sizes, test_mean + test_std, 
                 test_mean - test_std, alpha=0.15, color='b')
plt.grid()
plt.title('Curva de aprendizaje')
plt.legend(loc='upper right')
plt.xlabel('Cant de ejemplos de entrenamiento')
plt.ylabel('exactitud')
plt.show()


# PREDECIR BENIGNO

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
          'Bwd_Pkt_Len_Mean','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Output'))
X_test.loc[0] = (443,6,9,7,553,3773,61.44444444,539,63.65597482,49.51020264,553,0)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOT

X_test = pd.DataFrame(columns=('Dst_Port','Protocol','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean',
          'Bwd_Pkt_Len_Mean','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Output'))
X_test.loc[0] = (8080,6,3,4,326,129,108.6666667,32.25,257.2678158,343.0237544,326,1)
y_pred = classifier.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = classifier.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


