corpus = open('Dataset.txt').read()
docs = corpus.split('\n')
X, y = [], []
for doc in docs:
    i, l = doc.split(':')
    X.append(i)
    y.append(l)

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)




from sklearn.model_selection import train_test_split
## random_State also parameter for static the train Data
train_X, test_X, train_y, test_y = train_test_split(matrix_X, y, train_size = 0.8, shuffle = True)




print("Train X :",train_X)
print("Train y : ",train_y)
print("Test X :",test_X)




knn.fit(train_X, train_y)
print(knn.predict(test_X))
print(test_y)



