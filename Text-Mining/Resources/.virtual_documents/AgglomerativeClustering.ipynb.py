corpus = open('dataset.txt').read()
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



