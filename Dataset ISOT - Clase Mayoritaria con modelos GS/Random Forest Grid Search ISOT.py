# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:08:31 2020

@author: Sony
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

os.chdir("C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data")

df = pd.read_csv('Clase_Mayoritaria.csv', keep_default_na=False, na_values=[""])
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

df1 = df[['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]   

features =  ['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
             'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
             'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']

y=df.Output
print(y)


df.shape
df.dtypes

print("CANTIDAD DE DATOS PARA CADA CLASE: ")
print(df.groupby('Output').size())

X_train, X_test, y_train, y_test = train_test_split( df1, y, test_size = 0.2,
                                                    random_state =0 )

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 12,max_features=20,random_state = 0,class_weight='balanced')
max_depth=[20,35,10]
min_samples_split=[10,20,2]
min_samples_leaf=[10,20,1]
parameters = dict(max_depth=max_depth,min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
clf = GridSearchCV(model, parameters, cv = 5)

clf.fit(X_train,y_train)

clf.best_score_

clf.best_estimator_

clf.best_params_