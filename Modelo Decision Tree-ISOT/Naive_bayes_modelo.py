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
os.getcwd()

os.chdir("C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data")

df = pd.read_csv('capturas_4_2.csv', keep_default_na=False, na_values=[""])
df.head(10)
df1 = df[['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]   

features =  ['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']


y=df.Output

#DIVIDIR EL DATASET EN CONJUNTO DE ENTRENAMIENTO Y CONJUNTO DE PRUEBA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1, y,test_size=0.4,random_state = 0)

start = time.time()
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
model=  GaussianNB(priors=None, var_smoothing=1e-20)
model.fit(X_train,y_train)
end = time.time()
print ("Naive Bayes", end - start)
# Predicting the Test set results
y_pred = model.predict(X_test)



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

#------------------------------REPORTE DE RESULTADOS POR CLASE-------------------------------
print(classification_report(y_test,y_pred))

print("PRECISIÓN PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET ", metrics.precision_score(y_test,y_pred,average=None)*100)

print("RECALL PARA DETECTAR DIFERENTES MUESTRAS DE BOTNET: ", metrics.recall_score(y_test, y_pred, average=None)*100)

print ("EXACTITUD DEL MODELO:  ", sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)*100)

 # PREDECIR BOTNET BLACKOUT:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (64181,53,17,205,1,1,38,122,38,38,38,122,780487.8049,9756.097561,4878.04878,4878.04878,19,61,0,0,0)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET BLUE:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (54301,53,17,217,1,1,34,118,34,34,34,118,700460.8295,9216.589862,4608.294931,4608.294931,17,59,0,0,1)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET LIPHYRA:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (62238,53,17,246,1,1,37,121,37,37,37,121,642276.4228,8130.081301,4065.04065,4065.04065,18,60,0,0,2)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET BLACK ENERGY:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (52799,53,17,290,1,1,32,116,32,32,32,116,510344.8276,6896.551724,3448.275862,3448.275862,16,58,0,0,3)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET ZEUS:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (57366,53,17,216,1,1,34,118,34,34,34,118,703703.7037,9259.259259,4629.62963,4629.62963,17,59,0,0,4)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET ZYKLON:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (64628,53,17,260,1,1,36,120,36,36,36,120,600000,7692.307692,3846.153846,3846.153846,18,60,0,0,5)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))  
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))

# PREDECIR BOTNET CITADEL:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (63587,53,17,370,1,1,37,121,37,37,37,121,427027.027,5405.405405,2702.702703,2702.702703,18,60,0,0,6)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))


# PREDECIR BOTNET CITADEL2:
X_test = pd.DataFrame(columns=('Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts','Output'))

X_test.loc[0] = (55399,53,17,85,1,1,37,83,37,37,37,83,1411764.706,23529.41176,11764.70588,11764.70588,18,41,0,0,7)
y_pred = model.predict(X_test.drop(['Output'], axis = 1))
print("Prediccion: " + str(y_pred))
y_proba = model.predict_proba(X_test.drop(['Output'], axis = 1))
print("Probabilidad de Acierto: " + str(np.round(np.asarray(y_proba[0][y_pred])* 100, 2)))
