# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 18:31:14 2020

@author: Dr. Taimoor
"""

#Reading the data
corpus = open('dataset.txt').read()
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

# centroids =  nnc.cluster_centers
# print(centroids)
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

# print(matrix_input)

#importing KMeans and training model
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2)
# print(km)
km.fit(matrix_input[:-1])

centroids = km.cluster_centers_
print(centroids)

cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]

# #printing cluster labels of documents in the training data
# print(km.labels_)

# #printing label for the unseen document
# print(km.predict(matrix_input[-1:]))



