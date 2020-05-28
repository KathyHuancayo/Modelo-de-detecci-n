# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:56:50 2020

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

os.chdir("C://Users//Sony//Desktop//TESIS 2//CIC_AWS_Filtrado")

df = pd.read_csv('capturas_4_2.csv', keep_default_na=False, na_values=[""])
df.head(10)
#PREPARAR LOS DATOS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
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


df.shape
df.dtypes

print("CANTIDAD DE DATOS PARA CADA CLASE: ")
print(df.groupby('Output').size())

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_features=20,random_state = 0)
max_depth=[20,30,14]
min_samples_split=[10,20,2]
min_samples_leaf=[10,20,2]
parameters = dict(max_depth=max_depth,min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
clf = GridSearchCV(model, parameters, cv = 2)

clf.fit(df1, y)

clf.best_score_

clf.best_estimator_

clf.best_params_