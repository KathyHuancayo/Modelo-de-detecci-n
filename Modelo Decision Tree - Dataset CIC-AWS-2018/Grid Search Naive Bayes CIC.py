# -*- coding: utf-8 -*-
"""
Created on Fri May 22 03:04:21 2020

@author: Sony
"""


#PROCESAMIENTO
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//CIC_AWS_Filtrado.csv')

df=pd.DataFrame(datos)

X = datos[['Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
model=  GaussianNB()
var_smoothing=[1e-20,1e-10,1e-5]
parameters = dict(var_smoothing=var_smoothing)
clf = GridSearchCV(model, parameters, cv = 5)

clf.fit(X, y)

clf.best_score_

clf.best_estimator_

clf.best_params_