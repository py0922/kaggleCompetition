#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:43:52 2019

@author: Emma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:32:18 2019

@author: Emma
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def pre_data(train_file,test_file):
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)


    prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
    #prices.hist()

#smoothing data
    y_train = np.log1p(train_df.pop('SalePrice'))


#combine train data and test data
    all_df = pd.concat((train_df, test_df), axis=0)

#    all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
    #print(pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())

#turn no numeric data into one-hot category
    all_dummy_df = pd.get_dummies(all_df)
#    all_dummy_df.to_csv('temp.csv',sep=',',index=False, header=True)


# fill NA data with mean value
    mean_cols = all_dummy_df.mean()
    all_dummy_df = all_dummy_df.fillna(mean_cols)


#normilize numeric data
    numeric_cols = all_df.columns[all_df.dtypes != 'object']
    numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
    numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
    all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


#split test data and train data
    dummy_train_df = all_dummy_df.loc[train_df.index]
    dummy_test_df = all_dummy_df.loc[test_df.index]
    
    return dummy_train_df,y_train,dummy_test_df




if __name__=='__main__':

    train_df,y_train,test_df=pre_data('train.csv','test.csv')
    X_train = train_df.values
    X_test = test_df.values

    alphas = np.logspace(-3, 2, 50)

#choose the best parameter for RandomForestRegressor
#    test_scores = []
#    for alpha in alphas:
#        clf = Ridge(alpha)
#        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#        test_scores.append(np.mean(test_score))
#    
#
#    plt.plot(alphas, test_scores)
#    plt.title("Alpha vs CV Error")

#choose the best parameter for RandomForestRegressor
#    max_features = [.1, .3, .5, .7, .9, .99]
#    test_scores = []
#    for max_feat in max_features:
#        clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
#        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
#        test_scores.append(np.mean(test_score))
#    
#    plt.plot(max_features, test_scores)
#    plt.title("Max Features vs CV Error");

#use two model to predict
#    ridge = Ridge(alpha=15)
#    rf = RandomForestRegressor(n_estimators=500, max_features=.3)
#
#    ridge.fit(X_train, y_train)
#    rf.fit(X_train, y_train)
#
#
#    y_ridge = np.expm1(ridge.predict(X_test))
#    y_rf = np.expm1(rf.predict(X_test))
#
#    y_final = (y_ridge + y_rf) / 2
#    
#
#    submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})
#
#    submission_df.to_csv('submissions.csv', sep=',',index=False, header=True)


