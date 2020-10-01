# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:48:21 2020

@author: Admin
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('claims_paid.csv')
features=data.iloc[:,0:-1].values
labels=data.iloc[:,-1].values

data.isnull().sum()

"""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cd=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
features=np.array(cd.fit_transform(features),dtype=np.str)
"""



from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression
poly=PolynomialFeatures(degree=5)
features_poly=poly.fit_transform(features)
regressor=LinearRegression()
regressor.fit(features_poly,labels)

plt.scatter(features,labels,color='red')
plt.plot(features,regressor.predict(features_poly),color='blue')
plt.title('predicted graph')
plt.xlabel('year')
plt.ylabel('cost')

regressor.predict(poly.transform([[1981]]))


