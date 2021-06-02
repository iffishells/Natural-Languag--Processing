# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:08:10 2019

@author: taimo
"""

corpus = open('dataset.txt').read()
docs = corpus.split('\n')
X = []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())

## ready the input format
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

## inputing into the model
from sklearn.cluster import AgglomerativeClustering
aggClus = AgglomerativeClustering()
aggClus.fit(matrix_X.toarray())
print(aggClus.labels_)
