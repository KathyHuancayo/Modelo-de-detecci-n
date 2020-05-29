# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:46:40 2020

@author: Sony
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
 
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
os.getcwd()
datos = pd.read_csv('C://Users//Sony//Desktop//TESIS 2//isot_app_and_botnet_dataset//botnet_data//capturas_4_2.csv')

df=pd.DataFrame(datos)

X = datos[['Src_Port','Dst_Port','Protocol','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts',
          'TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Mean','Fwd Pkt Len Max','Fwd Pkt Len Min',
          'Bwd_Pkt_Len_Mean','Flow_Byts/s','Flow_Pkts/s','Fwd_Pkts/s','Bwd_Pkts/s','Subflow_Fwd_Byts','Subflow_Bwd_Byts','Subflow_Bwd_Pkts','Subflow_Fwd_Pkts']]  

y=datos['Output']

from sklearn.model_selection import GridSearchCV
model = KNeighborsClassifier(metric='minkowski')
n_neighbors =[4,10,1]
leaf_size=[30,50,70]
algorithm=['kd_tree','ball_tree','brute']
parameters = dict(n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size)
clf = GridSearchCV(model, parameters, cv = 2)

clf.fit(X,y)

clf.best_estimator_

clf.best_score_

clf.best_params_

