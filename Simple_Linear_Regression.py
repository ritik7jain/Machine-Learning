#Simple Linear Regression

#Importing Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#Importing Dataset
df=pd.read_csv("Salary_Data.csv")
features = df.iloc[:,:-1].values
labels=df.iloc[:,-1].values

#Checking nan values
df.isnull().sum()

#Splitting data into training and testing dataset
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=0)

#Applying Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#Predicting the values
labels_pred=regressor.predict(features_test)
regressor.predict([[6.9]])
regressor.predict(np.array(6.9).reshape(1,1))

#Checking accuracy score
Score = regressor.score(features_train, labels_train)
Score = regressor.score(features_test, labels_test)

#Evaluating the training set results
plt.scatter(features_train,labels_train,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title('Salary v/s Experience')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

# Evaluating for test set results
plt.scatter(features_test,labels_test,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title('Salary v/s Experience')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()
