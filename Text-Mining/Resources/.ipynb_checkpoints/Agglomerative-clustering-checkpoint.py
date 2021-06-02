# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:29:25 2020

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

#Importing agglomerative clustering and training the model
from sklearn.cluster import AgglomerativeClustering
aggClus = AgglomerativeClustering()
aggClus.fit(matrix_input.toarray())

#printing labels for the input documents that are clustered
print('doc 1 - 5 labels: ', aggClus.labels_)

#the function predict is not defined for agglomerative clustering