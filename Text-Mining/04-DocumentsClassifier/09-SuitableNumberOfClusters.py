# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 01:49:13 2019

@author: Taimoor
"""

corpus = open('dataset.csv').read()
docs = corpus.split('\n')
docs.remove(docs[0])
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(docs)
from sklearn.cluster import KMeans

km1 = KMeans(n_clusters = 1)
km2 = KMeans(n_clusters = 2)
km3 = KMeans(n_clusters = 3)
km4 = KMeans(n_clusters = 4)

km1.fit(matrix_X)
km2.fit(matrix_X)
km3.fit(matrix_X)
km4.fit(matrix_X)

print(km1.inertia_)
print(km2.inertia_)
print(km3.inertia_)
print(km4.inertia_)


# inertia_ is actually the points ont the curve which squard Error in the number given cluster