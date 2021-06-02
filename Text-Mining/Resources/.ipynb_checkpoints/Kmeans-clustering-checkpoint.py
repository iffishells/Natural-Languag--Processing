# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:22:32 2020

@author: Dr. Taimoor
"""
#Reading the input data
corpus = ['milk bread bread bread', 
          'break milk milk bread',
         'milk milk milk bread bread bread bread',
         'cat cat cat dog dog bark',
         'dog dog cat bark mew mew',
         'cat dog cat dog mew']

#Structuring input data
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
matrix_input = tfidf.fit_transform(corpus)

print(matrix_input)

#importing KMeans and training model
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
print(km)
km.fit(matrix_input[:-1])

#printing cluster labels of documents in the training data
print(km.labels_)

#printing label for the unseen document
print(km.predict(matrix_input[-1:]))