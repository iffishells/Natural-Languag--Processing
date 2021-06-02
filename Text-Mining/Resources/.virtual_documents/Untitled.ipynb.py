import pandas as pd
corpus = pd.read_csv('dataset-CalheirosMoroRita-2017.csv', encoding = 'Latin-1', delimiter = '\n')
print('Corpus as Dataframe', corpus)

data = corpus.values
print('Data as array of arrays of string', data)

clean_data = [data[i][0] for i in range(0, len(data))]
print('Clean data as Array of strings', clean_data)

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_input = vec.fit_transform(clean_data)

print('Structured input data', matrix_input)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 5)
km.fit(matrix_input)

print('Cluster labels of documents', km.labels_)



