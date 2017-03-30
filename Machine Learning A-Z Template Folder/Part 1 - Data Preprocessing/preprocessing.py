# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:08:03 2017

@author: madha
"""

# Data Pre-Processing

#Importing libraries
import numpy as np # mathematical tools 
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
                
from sklearn.preprocessing import Imputer
                
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

#categorical variable handling

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
labelencoder_Y.fit_transform(y)
y = labelencoder_Y.fit_transform(y)

#splitting the dataset into the training set and test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0) 

# feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train =sc_X.fit_transform(X_train)
Y_test =sc_X.fit_transform(X_test)
