#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:05:57 2019

@author: Emma
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

data = pd.read_csv('input/Combined_News_DJIA.csv')


data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


feature_extraction = TfidfVectorizer()

#IF-IDF
X_train = feature_extraction.fit_transform(train["combined_news"].values)
X_test = feature_extraction.transform(test["combined_news"].values)


y_train = train["Label"].values

y_test = test["Label"].values


clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)

predictions = clf.predict_proba(X_test)
#AUC-roc
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))