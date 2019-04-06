#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:38:26 2019

@author: Emma
"""

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
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem import WordNetLemmatizer
stop = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

#filter numbers
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

#filter stop words and numbers
def check(word):
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    else:
        return True



if __name__=='__main__':
    data = pd.read_csv('input/Combined_News_DJIA.csv')


    data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    
#    lowcase and delete ' and "
    X_train = train["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()
    X_test = test["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()

#delete stop words and number and proto-word
    X_train = X_train.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
    X_test = X_test.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])

# convert into string because sklearn's input is string
    X_train = X_train.apply(lambda x: ' '.join(x))
    X_test = X_test.apply(lambda x: ' '.join(x))

# IF-IDF
    feature_extraction = TfidfVectorizer(lowercase=False)  
    X_train = feature_extraction.fit_transform(X_train.values)
    X_test = feature_extraction.transform(X_test.values)
    
    
    clf = SVC(probability=True, kernel='rbf')
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))
    