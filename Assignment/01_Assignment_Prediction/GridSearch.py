#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 06:01:42 2021

@author: shohanursobuj
"""

from IPython import get_ipython
get_ipython().magic('reset -sf') 

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 


clasificators = [LinearSVC(), DecisionTreeClassifier(), RandomForestClassifier()]


clasificatorsString = ['LinearSVC()', 'DecisionTreeClassifier()',  
                 'RandomForestClassifier()']

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

X_train, X_test, y_train, y_test = train_test_split(X_public, y_public, 
                                    test_size=0.25, random_state=0)

parameters0 = {'max_iter': np.arange(1,1000, 200),
               'intercept_scaling': np.arange(1,100,30)}
parameters1 = {'criterion': ['gini', 'entropy'],
               'max_features': np.arange(1,100,10),
               'max_depth': np.arange(1,500,50),
               'min_samples_split': np.arange(1,10,3)}
parameters2 = {'n_estimators': np.arange(400,403,1),
               'criterion': ['gini', 'entropy'],
               'max_features': np.arange(20,23,1),
               'max_depth': np.arange(450,453,1)}


parameters = [parameters0, parameters1, parameters2]
for i in range(0,3):
    print('-----------------------------------------')
    print('Testing: ' + '[' + str(i) + ']' + ' ' + str(clasificatorsString[i]))
    print('-----------------------------------------')
    print('In progress...')
    clf = clasificators[i]
    grid = GridSearchCV(clf, parameters[i])
    grid.fit(X_train,y_train)
    
    print('The best parameters combination is : ')
    print(grid.best_params_)
    print('The best accuracy via grid is: ')
    print(grid.best_score_)
    
    y_predicted = grid.predict(X_test)
    print('_________________')
    print('Final prediction: ')
    print(roc_auc_score(y_predicted, y_test))
    print(' ')
    
    
