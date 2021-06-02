# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:31:14 2020

@author: Dr. Taimoor
"""

#Reading the data
corpus = open('D:\\Dataset.txt').read()
docs = corpus.split('\n')

#Structuring input data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
matrix_input = tfidf.fit_transform(docs)
print(matrix_input)

#Importing NearestNeighbors and Training model
from sklearn.neighbors import NearestNeighbors
nnc = NearestNeighbors()
nnc.fit(matrix_input)

#printing nearest neighbors to the first document and their respective distances
dist, neighbors = nnc.kneighbors(matrix_input[0], 3)
print('neighbors', neighbors)
print('distances', dist)

#printing with the first document i.e., the reference document itself ignored
print('neighbors', neighbors[0][1:])
print('distances', dist[0][1:])

#printing nearest neighbors that lie within the given radius of the reference document
dist, neighbors = nnc.radius_neighbors(matrix_input[3], radius = 1.5)
print('radius based neighbors: ', neighbors[1:])
print('radius based neighbors distances', dist[1:])