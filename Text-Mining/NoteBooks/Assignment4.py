# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:22:32 2019

@author: Taimoor
"""

corpus = open('E:\\__sentiDataset.txt').read()
docs = corpus.split('\n')
domain, label, rating, X = [], [], [], []
for d in docs:
    dm, lb, rt, inp = d.split('\t')
    domain.append(dm)
    label.append(lb)
    rating.append(int(rt))
    X.append(inp)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(max_features = 200, ngram_range = (1, 3))
matrix_X = vec.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

model1 = KNeighborsClassifier(n_neighbors = 3)
model2 = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
model3 = MultinomialNB(alpha = 0.1, fit_prior = False)
model4 = DecisionTreeClassifier(max_depth = 3)
model5 = DecisionTreeClassifier(max_depth = 4)

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

from sklearn.metrics import precision_score, recall_score, f1_score

m1p, m1r, m1f = 0, 0, 0
m2p, m2r, m2f = 0, 0, 0
m3p, m3r, m3f = 0, 0, 0
m4p, m4r, m4f = 0, 0, 0 
m5p, m5r, m5f = 0, 0, 0

import numpy as np
rating = np.array(rating)

for train_ids, test_ids in kf.split(matrix_X):
    train_X, test_X = matrix_X[train_ids], matrix_X[test_ids]
    train_y, test_y = rating[train_ids], rating[test_ids]
    #training models
    model1.fit(train_X, train_y)
    model2.fit(train_X, train_y)
    model3.fit(train_X, train_y)
    model4.fit(train_X, train_y)
    model5.fit(train_X, train_y)
    #testing models
    py1 = model1.predict(test_X)
    py2 = model2.predict(test_X)
    py3 = model3.predict(test_X)
    py4 = model4.predict(test_X)
    py5 = model5.predict(test_X)
    #evaluate
    m1p += precision_score(test_y, py1, average = 'macro')
    m2p += precision_score(test_y, py2, average = 'macro')
    m3p += precision_score(test_y, py3, average = 'macro')
    m4p += precision_score(test_y, py4, average = 'macro')
    m5p += precision_score(test_y, py5, average = 'macro')
    
    m1r += recall_score(test_y, py1, average = 'macro')
    m2r += recall_score(test_y, py2, average = 'macro')
    m3r += recall_score(test_y, py3, average = 'macro')
    m4r += recall_score(test_y, py4, average = 'macro')
    m5r += recall_score(test_y, py5, average = 'macro')
    
    m1f += f1_score(test_y, py1, average = 'macro')
    m2f += f1_score(test_y, py2, average = 'macro')
    m3f += f1_score(test_y, py3, average = 'macro')
    m4f += f1_score(test_y, py4, average = 'macro')
    m5f += f1_score(test_y, py5, average = 'macro')

print('model 1... (5-fold cross validation)...')
print('precision score: ', m1p/5.0)
print('recall score: ', m1r/5.0)
print('f1 score: ', m1f/5.0)

print('model 2... (5-fold cross validation)...')
print('precision score: ', m2p/5.0)
print('recall score: ', m2r/5.0)
print('f1 score: ', m2f/5.0)

print('model 3... (5-fold cross validation)...')
print('precision score: ', m3p/5.0)
print('recall score: ', m3r/5.0)
print('f1 score: ', m3f/5.0)

print('model 4... (5-fold cross validation)...')
print('precision score: ', m4p/5.0)
print('recall score: ', m4r/5.0)
print('f1 score: ', m4f/5.0)

print('model 5... (5-fold cross validation)...')
print('precision score: ', m5p/5.0)
print('recall score: ', m5r/5.0)
print('f1 score: ', m5f/5.0)