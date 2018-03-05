path = "D:\/backup\Python assignment\House Prices Advanced Regression Techniques\/train.csv"

path_test = "D:\/backup\Python assignment\House Prices Advanced Regression Techniques\/test.csv"

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge

from sklearn.svm import SVR, LinearSVR

import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

import xgboost as xgb

train_data =pd.read_csv(path)

test_data = pd.read_csv(path_test)

model = Lasso(alpha = 50, tol = 1e-6)

Target_SalePrice = train_data['SalePrice']




corr_varialbe = train_data.corr() #Find out the top 10 most correlated variables to 'SalePrice'. Use these variables to build the model

corr_varialbe_10 = corr_varialbe.nlargest(11, 'SalePrice')['SalePrice']

high_corr_variable_name = corr_varialbe_10.index.astype(basestring)

print high_corr_variable_name


high_corr_variable = train_data[high_corr_variable_name] #choose the top 10 most correlated variables columns as training set

high_corr_variable = high_corr_variable.drop('SalePrice', axis = 1)

test_data = test_data[high_corr_variable_name[1:]] #Choose the corresponding columns in the test set

#test_data = test_data[test_data['GarageCars'].notnull() & test_data['TotalBsmtSF'].notnull()]

test_data = test_data.fillna(0.1) #fill the Na data


#split the training set into two parts. One part is for training and the other is for validation
high_corr_variable_train_x, high_corr_variable_test_x, high_corr_variable_train_y, high_corr_variable_test_y = train_test_split(high_corr_variable,Target_SalePrice, test_size = 0.2, random_state=7 ) 

print high_corr_variable_train_x.shape

print high_corr_variable_test_x.shape

print high_corr_variable_train_y.shape

print high_corr_variable_test_y.shape


# The following codes illustrate the use of KFold

# ntrain = high_corr_variable.shape[0]

# ntest = test_data.shape[0]

# kf = KFold(ntrain, n_folds = 5, random_state = 0) 

# print kf

# for i, (train_index, test_index) in enumerate(kf):

# 	print i, len(train_index), len(test_index)

# 	print len(train_index) + len(test_index)



class SklearnExtension(object): #This makes it more handy to fit model

	def __init__(self, clf, *args):

		#params['random_state'] = seed

		self.clf = clf()


	def train(self, x_train, y_train):

		self.clf.fit(x_train, y_train)


	def predict(self, x):

		return self.clf.predict(x)

	def fit(self, x, y):

		return self.clf.fit(x, y)

	def feature_importances(self, x, y):

		print (self.clf.fit(x, y).feature_importances_)


lg = SklearnExtension(clf = LogisticRegression)

lr = SklearnExtension(clf = LinearRegression)

Rg = SklearnExtension(clf = Ridge)

SR = SklearnExtension(clf = SVR)

la = SklearnExtension(clf = Lasso)


#build the model with splitted data in the training set

lg.train(high_corr_variable_train_x, high_corr_variable_train_y)

lr.train(high_corr_variable_train_x, high_corr_variable_train_y)

Rg.train(high_corr_variable_train_x, high_corr_variable_train_y)

SR.train(high_corr_variable_train_x, high_corr_variable_train_y)

la.train(high_corr_variable_train_x, high_corr_variable_train_y)


# predict the sale price in trianing set
lg_high_corr_test_y_predict = pd.DataFrame(lg.predict(high_corr_variable_test_x), columns = ['lg']) 

lr_high_corr_test_y_predict = pd.DataFrame(lr.predict(high_corr_variable_test_x), columns = ['lr'])

Rg_high_corr_test_y_predict = pd.DataFrame(Rg.predict(high_corr_variable_test_x), columns = ['Rg'])

SR_high_corr_test_y_predict = pd.DataFrame(SR.predict(high_corr_variable_test_x), columns = ['SR'])

la_high_corr_test_y_predict = pd.DataFrame(la.predict(high_corr_variable_test_x), columns = ['la'])



# predict the sale price in test set (Real test data)
lg_test_data_y_predict = pd.DataFrame(lg.predict(test_data), columns = ['lg'])
lr_test_data_y_predict = pd.DataFrame(lg.predict(test_data), columns = ['lr'])
Rg_test_data_y_predict = pd.DataFrame(lg.predict(test_data), columns = ['Rg'])
SR_test_data_y_predict = pd.DataFrame(lg.predict(test_data), columns = ['SR'])
la_test_data_y_predict = pd.DataFrame(lg.predict(test_data), columns = ['la'])

#print lg_test_data_y_predict



Xgboost_train_x = pd.concat([lg_high_corr_test_y_predict, lr_high_corr_test_y_predict, Rg_high_corr_test_y_predict, SR_high_corr_test_y_predict, la_high_corr_test_y_predict], axis =1)


Xgboost_train_y = high_corr_variable_test_y


Xgboost_test_x = pd.concat([lg_test_data_y_predict, lr_test_data_y_predict, Rg_test_data_y_predict, SR_test_data_y_predict, la_test_data_y_predict], axis =1)

#print Xgboost_train_x.head()


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(Xgboost_train_x, Xgboost_train_y)



# predictions = pd.DataFrame(gbm.predict(Xgboost_test_x))

# predictions.to_csv('Prediction result.csv')



#This section is using joblib to save the trained model for future use

finalized_model_joblib = 'finalized_model_joblib.sav'

joblib.dump(gbm, finalized_model_joblib)

loaded_model_from_joblib = joblib.load(finalized_model_joblib)

predictions_joblib = pd.DataFrame(loaded_model_from_joblib.predict(Xgboost_test_x))

predictions_joblib.to_csv('Prediction result_from joblib.csv')