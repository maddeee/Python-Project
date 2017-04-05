# multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]
                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predication the test set

y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)), values=X, axis=1)

X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

#step 2
X_opt = X[:,[0,1,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

#step 3

X_opt = X[:,[0,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

#step  4

#step 3

X_opt = X[:,[0,3,5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()



