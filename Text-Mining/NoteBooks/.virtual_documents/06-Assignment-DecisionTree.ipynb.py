corpus  = open("Datasets/badges.data").read()
from sklearn.tree import DecisionTreeClassifier


docs = corpus.split('\n')
X, y = [], [] 
for doc in docs:
    l = doc[:1]
    i = doc[2:]
    X.append(i)
    y.append(l)


from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
matrix_X = vec.fit_transform(X)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(matrix_X[:290], y[:290])



#predicted labels of the last four documents
print(knn.predict(matrix_X[293])) 
#prediction probability of the two labels for each of the last four documents
print(knn.predict_proba(matrix_X[290])) 


corpus  = open("Datasets/badges.data").read()
from sklearn.tree import DecisionTreeClassifier

docs = corpus.split("\n") ## split by on the line termination


x  = []  ## name of the people
y  = []  ## name of the badges

for doc in docs:
    
    l = doc[:1]  # badges
    
    i = doc[2:]  ## name of the poeple
    
    x.append(i)
    y.append(l)



from sklearn.feature_extraction.text import TfidfVectorizer

vec  = TfidfVectorizer()

matrix_x = vec.fit_transform(x)


# matrix_x.toarray()


from sklearn.tree import DecisionTreeClassifier
dtc  = DecisionTreeClassifier(max_depth=5)






dtc.fit(matrix_x[:284],y[:284]) 
print("Decision Tree Classifier , label "+ str(dtc.predict(matrix_X[291])))
print("Decision Tree Classifier , prob "+ str(dtc.predict_proba(matrix_X[291])))


help(Des)



