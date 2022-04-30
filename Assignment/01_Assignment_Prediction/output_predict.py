#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 06:01:42 2021

@author: shohanursobuj
"""


from IPython import get_ipython
get_ipython().magic('reset -sf') 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 

X_public = DataFrame(np.load('X_public.npy',allow_pickle=True)).drop_duplicates()
X_eval = DataFrame(np.load('X_eval.npy',allow_pickle=True)).drop_duplicates()
y_public = (np.load('y_public.npy',allow_pickle=True).ravel())

for i in range(180,200):
    X_public[i] = LabelEncoder().fit_transform(X_public[i])
    X_eval[i] = LabelEncoder().fit_transform(X_eval[i])
    

X_public_imputed = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(X_public)
X_eval_imputed = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(X_eval)

X_public_scaled = StandardScaler().fit_transform(X_public_imputed)
X_eval_scaled = StandardScaler().fit_transform(X_eval_imputed)

X_public_selected = VarianceThreshold(threshold=0.8).fit_transform(X_public_scaled)
X_eval_selected = VarianceThreshold(threshold=0.8).fit_transform(X_eval_scaled)

X_public = DataFrame(X_public_selected)
X_eval = DataFrame(X_eval_selected)

#imp = SimpleImputer(missing_values='NaN')
#X_public = imp.fit_transform(X_public)

X_train, X_test, y_train, y_test = train_test_split(X_public, y_public, 
                                    test_size=0.25, random_state=0)

clf = QuadraticDiscriminantAnalysis(reg_param = 0.6)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(roc_auc_score(y_predict, y_test))

y_predicted = clf.predict(X_eval)
np.save('y_predikcia.npy', y_predicted)
