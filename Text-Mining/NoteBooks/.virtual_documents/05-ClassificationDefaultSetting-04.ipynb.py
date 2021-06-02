corpus = open("Dataset.txt").read()


docs = corpus.split('\n')


docs


X ,Y = [] , []
# separating the data into docs and lebels
for doc in docs:
    i , l = doc.split(":")
    X.append(i.strip())
    Y.append(l.strip())
    


X #docs


Y # labels


## strucutre the input data

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer() 
matrix_X = vec.fit_transform(X)


matrix_X.toarray()



vec.vocabulary_ 
# milk : 5 "5" is the index in the array and milk is that quantiy which lie on
# on the column


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(matrix_X[:5],Y[:5])
print("KNN classifier label : " + str(knn.predict(matrix_X[5])))
print("KNN classifier Prob  : " + str(knn.predict_proba(matrix_X[5])))


from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()
nbc.fit(matrix_X[:5],Y[:5])

print("Naive bayes Classifier ,Label : "+str(nbc.predict(matrix_X[5])))
print("Naive bayes Classifier ,prob : "+str(nbc.predict_proba(matrix_X[5])))



from sklearn.tree import DecisionTreeClassifier
dtc  = DecisionTreeClassifier()
dtc.fit(matrix_X[:5],Y[:5])
print("Decision Tree Classifier , label "+ str(dtc.predict(matrix_X[5])))
print("Decision Tree Classifier , prob "+ str(dtc.predict_proba(matrix_X[5])))


from sklearn.linear_model import SGDClassifier

lc = SGDClassifier()
lc.fit(matrix_X[:5],y[:5])
print("Linear Classifer ,label : "+str(lc.predict(matrix_X[5])))
