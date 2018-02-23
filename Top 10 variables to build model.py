path = "D:\/backup\Python assignment\House Prices Advanced Regression Techniques\/train.csv"

path_test = "D:\/backup\Python assignment\House Prices Advanced Regression Techniques\/test.csv"

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats


train_data =pd.read_csv(path)

test_data = pd.read_csv(path_test)

model = LogisticRegression()

Target_SalePrice = train_data['SalePrice']


corr_varialbe = train_data.corr()

corr_varialbe_10 = corr_varialbe.nlargest(11, 'SalePrice')['SalePrice']

#print corr_varialbe_10

high_corr_variable_name = corr_varialbe_10.index.astype(basestring)



#print type(high_corr_variable_name)


high_corr_variable = train_data[high_corr_variable_name]

high_corr_variable = high_corr_variable.drop('SalePrice', axis = 1)

test_data = test_data[high_corr_variable_name[1:]]

#test_data = test_data[test_data['GarageCars'].notnull() & test_data['TotalBsmtSF'].notnull()]

test_data = test_data.fillna(0.1)


# print high_corr_variable.info()

# print test_data.info()

# print test_data.shape

# print test_data.isnull().sum()




model.fit(high_corr_variable, Target_SalePrice)

y_test_predict = pd.DataFrame(model.predict(test_data))

print len(y_test_predict)

y_test_predict.to_csv('Prediction result.csv')

