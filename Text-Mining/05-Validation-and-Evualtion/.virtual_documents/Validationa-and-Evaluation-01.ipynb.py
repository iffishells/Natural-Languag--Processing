movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"


df  = pd.read_csv(movies_review_path)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import train_test_split


reviewDocs_x = df.iloc[:,0]
# reviewDocs_x


reviewDocs_y = df.iloc[:,1]
# reviewDocs_y


tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(reviewDocs_x)


# tfidf


# tfidf.toarray()


train_x,test_x,train_y,test_y = train_test_split(tfidf,
                                                reviewDocs_y,
                                                train_size=0.1
                                                 , 
                                                shuffle=True)


clf = MultinomialNB().fit(train_x, train_y)


pred_y = clf.predict(test_x)
pred_y


from sklearn.metrics import accuracy_score


accc = accuracy_score(test_y,pred_y)


print("Accuracy : ",accc)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import train_test_split # bcz Hold out appraoch

movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
 
reviewDocs_x = df.iloc[:,0]  ## x component

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(reviewDocs_x)  # model language

train_x,test_x,train_y,test_y = train_test_split(tfidf,
                                                reviewDocs_y,
                                                train_size=0.8
                                                 , 
                                                shuffle=True)  ## spliting data according to hold out cross validation

clf = MultinomialNB().fit(train_x, train_y)  ## put into the Classifer

pred_y = clf.predict(test_x)  # prediction
# pred_y

from sklearn.metrics import accuracy_score
accc = accuracy_score(test_y,pred_y)
print("Accuracy : ",accc)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import KFold  
from sklearn.metrics import accuracy_score


movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
 
x = df.iloc[:,0]  ## x component
print(x)
y = df.iloc[:,1]

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(x)  # model language

# ----------------------------------------------------------------------------
acc = []
kf  = KFold(n_splits=5)
for trian_ids ,test_ids in kf.split(tfidf,y):
    train_x , test_x = tfidf[trian_ids],tfidf[test_ids]
    
    train_y ,test_y = y[trian_ids] , y[test_ids]
    
    clf = MultinomialNB().fit(train_x, train_y)  ## put into the Classifer
    pred_y = clf.predict(test_x)
    acc.append(accuracy_score(test_y,pred_y))
    
    
     


print("Accuracy : ",acc)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import accuracy_score


movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
df =  df.head(500) 
x = df.iloc[:,0]  ## x component

y = df.iloc[:,1]

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(x)  # model language

# ----------------------------------------------------------------------------
acc = []
loo = LeaveOneOut() 
for trian_ids ,test_ids in loo.split(x,y):
    train_x , test_x = tfidf[trian_ids],tfidf[test_ids]
    
    train_y ,test_y = y[trian_ids] , y[test_ids]
    
    clf = MultinomialNB().fit(train_x, train_y)  ## put into the Classifer
    pred_y = clf.predict(test_x)
    acc.append(accuracy_score(test_y,pred_y))
    
    
     


print("Accuracy : ",sum(acc)/len(acc))



from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier  #update
from sklearn.model_selection import KFold
import  pandas as pd
# from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import accuracy_score


movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
df =  df.head(500)  ## capture limited data
x = df.iloc[:,0]  ## x component

y = df.iloc[:,1]

tfidf = TfidfVectorizer(min_df=20,
                        max_df=3000).fit_transform(x)  # model language

knn = KNeighborsClassifier(n_neighbors=1)  ## put into the Classifer
    
# ----------------------------------------------------------------------------
acc = []
kf = KFold(n_splits=5) 
score = 0 
for trian_ids ,test_ids in kf.split(tfidf):
    train_x , test_x = tfidf[trian_ids],tfidf[test_ids]
    
    train_y ,test_y = y[trian_ids] , y[test_ids]
    
    knn.fit(train_x, train_y)  ## put into the Classifer
    pred_y = knn.predict(test_x)
#     acc.append(accuracy_score(test_y,pred_y))
    score += accuracy_score(test_y,pred_y,normalize=True) 
    
    
     

print(score/5)




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import train_test_split # bcz Hold out appraoch

movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
 
reviewDocs_x = df.iloc[:,0]  ## x component

reviewDocs_y = df.iloc[:,1]  ## y component

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(reviewDocs_x)  # model language

train_x,test_x,train_y,test_y = train_test_split(tfidf,
                                                reviewDocs_y,
                                                train_size=0.8
                                                 , 
                                                shuffle=True)  ## spliting data according to hold out cross validation

clf = MultinomialNB().fit(train_x, train_y)  ## put into the Classifer

pred_y = clf.predict(test_x)  # prediction
# pred_y

from sklearn.metrics import accuracy_score , precision_score,f1_score,recall_score
accc = accuracy_score(test_y,pred_y)
pre_score = precision_score(test_y,pred_y , average="weighted")
f1Score = f1_score(test_y,pred_y, average="weighted")
recallScore = recall_score(test_y,pred_y, average="weighted")

print(" Accuracy Score {} \n Precision Score {} \n F1 Score {} \n Recall Score {} ".format(accc,pre_score,f1Score,recallScore))




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import train_test_split # bcz Hold out appraoch

movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
 
reviewDocs_x = df.iloc[:,0]  ## x component

reviewDocs_y = df.iloc[:,1]  ## x component

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(reviewDocs_x)  # model language

train_x,test_x,train_y,test_y = train_test_split(tfidf,
                                                reviewDocs_y,
                                                train_size=0.8
                                                 , 
                                                shuffle=True)  ## spliting data according to hold out cross validation

clf = MultinomialNB().fit(train_x, train_y)  ## put into the Classifer

pred_y = clf.predict(test_x)  # prediction
# pred_y

from sklearn.metrics import confusion_matrix
CFM = confusion_matrix(test_y,pred_y)

print(" Confusion Matrix \n {}   ".format(CFM))



from sklearn.feature_extraction.text import TfidfVectorizer ## Vectorizer
from sklearn.naive_bayes import MultinomialNB  ## classifer
from sklearn.tree import DecisionTreeClassifier ## classifer
import  pandas as pd
from sklearn.model_selection import KFold    ## cross validation
from sklearn.metrics import f1_score    ## evluation


movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
df = df.head(15000) 
x = df.iloc[:,0]  ## x component

y = df.iloc[:,1]  

tfidf = TfidfVectorizer(min_df=20,
                        max_df=300000).fit_transform(x)  # model language

# ----------------------------------------------------------------------------
# marjor task

Mnb = MultinomialNB()
dt = DecisionTreeClassifier(random_state=0,max_depth=3)
kf  = KFold(n_splits=5)


## Count 
Mnb_count = 0
dt_count = 0
for trian_ids ,test_ids in kf.split(tfidf,y): # y optional bcz already defined above
    train_x , test_x = tfidf[trian_ids],tfidf[test_ids]
    
    train_y ,test_y = y[trian_ids] , y[test_ids]
    
    Mnb.fit(train_x, train_y)  ## put into the Classifer
    Mnb_pred_y = Mnb.predict(test_x)
    Mnb_count += f1_score(test_y,Mnb_pred_y , average="micro")
    
    ## Decision Tree
    dt.fit(train_x, train_y)
    dt_pred_y = dt.predict(test_x)
    dt_count += f1_score(test_y,dt_pred_y , average = "micro")

    
Mnb_count=Mnb_count/5
dt_count = dt_count/5

print("F1 Score from MultinomiaNB {} \n F1 Score from Decision Tree {} ".format(Mnb_count,dt_count))


import pandas as pd

moview_review_data_text_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/dataset-CalheirosMoroRita-2017.csv"

df = pd.read_csv(moview_review_data_text_path,error_bad_lines=False ,encoding="unicode_escape").read()

df = df.T
df

## Underprocessing


## importing all need of lib
import pandas as pd
import pprint as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import  pandas as pd
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

## load file  
Assignment_text_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/__sentiDataset.txt"
from sklearn.metrics import accuracy_score , precision_score,f1_score,recall_score

df = open(Assignment_text_path).read()

df = df.split("\n")
# df = df[:30]

## lenght each document  
splitted_by_line  = df[0].split("\t")

df_dict = {
    "input" : [],
    "label":[]
    
}

for doc_index in range(0,len(df)):
    single_doc = df[doc_index].split("\t")
    df_dict["label"].append(single_doc[2])
    df_dict["input"].append(str(single_doc[3]))

## load into the DataFram 
data  = pd.DataFrame(df_dict)



## pre-processing of data


def remove_string_special_characters(s):
    s = str(s)  
    # removes special characters with ' '
    stripped = re.sub('[^a-zA-z\s]', '', s)
    stripped = re.sub('_', '', stripped)
#     stripped = re.sub('.', '', stripped)
       
    # Change any white space to one space
    stripped = re.sub('\s+', ' ', stripped)
      
    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped get_ipython().getoutput("= '':")
            return stripped.lower()

for index,row in data.iterrows():
#     print(row["input"])
    doc = remove_string_special_characters(row["input"]) 
#     print(doc)
    
    stop_words = set(stopwords.words('english'))  
  
    word_tokens = word_tokenize(example_sent)  
    
    filtered_sentence = []  

    for w in word_tokens:  
        if w not in stop_words:
            filtered_sentence.append(porter_stemmer.stem(w))  ## stemming
#     print(word_tokens)  
    row["input"] = str(filtered_sentence)
    
    
    
x = data.iloc[:,0]  ## x component
# print(x)
y = data.iloc[:,1]

# print(y)
tfidf = TfidfVectorizer(max_features=200 ,ngram_range= (2,2)).fit_transform(x)  # model language

# ----------------------------------------------------------------------------

Mnb = MultinomialNB()

dt = DecisionTreeClassifier(random_state=0,max_depth=3 )
dt_2 = DecisionTreeClassifier(random_state=0,max_depth=30)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh_2 = KNeighborsClassifier(n_neighbors=5)

kf  = KFold(n_splits=5)


## Count 
Mnb_count = 0
dt_count = 0
dt_2_count =0
neigh_count=0
neigh_2_count=0

for trian_ids ,test_ids in kf.split(tfidf,y): # y optional bcz already defined above
    train_x , test_x = tfidf[trian_ids],tfidf[test_ids]
    
    train_y ,test_y = y[trian_ids] , y[test_ids]
    
    Mnb.fit(train_x, train_y)  ## put into the Classifer
    Mnb_pred_y = Mnb.predict(test_x)
    Mnb_count += f1_score(test_y,Mnb_pred_y , average="micro")
    
    ## Decision Tree
    dt.fit(train_x, train_y)
    dt_pred_y = dt.predict(test_x)
    dt_count += f1_score(test_y,dt_pred_y , average = "macro")
    # model 2
    dt_2.fit(train_x, train_y)
    dt_2_pred_y = dt.predict(test_x)
    dt_2_count += f1_score(test_y,dt_pred_y , average = "micro")
    
   
    ## KNN Neighbour
    neigh.fit(train_x, train_y)
    neigh_pred_y = dt.predict(test_x)
    neigh_count += f1_score(test_y,dt_pred_y , average = "micro")
    
    neigh_2.fit(train_x, train_y)
    neigh_2_pred_y = dt.predict(test_x)
    neigh_2_count += f1_score(test_y,dt_pred_y , average = "macro")
    
    
    
Mnb_count=Mnb_count/5
dt_count = dt_count/5
neigh_count = neigh_count/5
neigh_2_count = neigh_2_count/5
dt_2_count = dt_2_count/5

print("F1 Score from MultinomiaNB {} \nF1 Score from Decision Tree (depth =3){} \nF1 Score from Decision Tree(depth=30) {}  \nF1 Score from K Neighbour Model(N=3) {} \nF1 Score from K Neighbour Model(N=5) {}" .format(Mnb_count,dt_count,dt_2_count,neigh_count,neigh_2_count))
    


import numpy as np
import re
import nltk
from nltk.corpus import stopwords
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

#reading file

EngStopWord = open("EnglishStopWords.txt").read()
EngStopWord= EngStopWord.split("\n")

filePathForPreProcessing = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/__sentiDataset.txt"
df = open(filePathForPreProcessing).read()


## removing Stop words from the file
# df = df[:1000]  ## tool small section



## converting into the lower Case:
df = df.lower()

## reomving white space 
df = df.strip()

## Removing punctuation
from string import punctuation as punc

for ch in df:
    if ch in punc:
        df.replace(ch,'')


df = df.split(" ")

print(" Before removing StopWord Len : ",len(df))

for word in df:
#     print(word)
    if word in EngStopWord:
        df.remove(word)
print(" After removing StopWord Len : ",len(df))



import numpy as np
import tkinter as tk
import pycountry 
import tkinter as tk  
from functools import partial  
   
def levenshtein(seq1 , seq2):
    
    size_x = len(seq1)+1
#     print("len seq1 : ",size_x)
    size_y = len(seq2)+1
#     print("len seq2 : ",size_y )
    matrix = np.zeros((size_x,size_y))
    
    ## inialization of row 0 to len(size_x)
    for x in range(size_x):
        matrix[x,0] = x
        
#     print("X \n" ,matrix)
    
    ## inialization of columns 0 to len(Seq2)
    for y in range(size_y):
        matrix[0,y] = y
        
#     print("Y \n",matrix)
    
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x,y] = min(matrix[x-1,y],matrix[x-1,y-1],matrix[x,y-1])
#                 print("X loop = ".join(x) , matrix)
            else:
                matrix[x,y] = min(
                matrix[x-1,y]+1,
                matrix[x-1,y-1]+1,
                matrix[x,y-1]+1
                )
#                 print("y = ",x ,"\n", matrix)
                
#     print("final : \n", matrix)
    
    return (matrix[size_x-1,size_y-1])
        
    
def Sort_Tuple(tup):  
      
    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (tup[j][1] > tup[j + 1][1]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup 

def function_Calls(label_result,target):
    target = str(target.get())
    
    ##import country from pycountry Python API
    country_name=[]
    for name in range(20):
        country = list(pycountry.countries)[name].name
        country_name.append(country)

    # compare with the target from the list of the name (Name can be random) 
    ## but am added all the country name
    
    edit_distance = []
    for name in country_name:
    #     distace = levenshtein("Apple" , name)
        edit_distance.append((name,int(levenshtein(target , name))))
      
    result = str(Sort_Tuple(edit_distance)[0:3])
    label_result.config(text="Result = get_ipython().run_line_magic("s"", " % result)")
    return
#     Sort_Tuple(edit_distance)[0:3]


root = tk.Tk()  
root.geometry('800x500')
root.resizable(0,0)
  
root.title('Levenshtein Distance')  
#    
number1 = tk.StringVar()    
  
labelNum1 = tk.Label(root, text="Target String").grid(row=1, column=0)  
   
labelResult = tk.Label(root) ## WILL HOLD THE RESULT OF OUTPUT  
  
labelResult.grid(row=9, column=1)  
  
entryNum1 = tk.Entry(root, textvariable=number1).grid(row=1, column=2)  
  

function_Calls = partial(function_Calls, labelResult, number1)  
  
buttonCal = tk.Button(root, text="Submit", command=function_Calls).grid(row=3, column=0)  
  
root.mainloop()  



get_ipython().getoutput("pip install pycountry")


# import the enchant module
import enchant

# determining the values of the parameters
string1 = "Hello World"
string2 = "Hello d"

# the Levenshtein distance between
# string1 and string2
print(enchant.utils.levenshtein(string1, string2))



get_ipython().getoutput("pip install enchant")



import numpy as np

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])


levenshtein("Apple","Aple")


import nltk

word_data = "<s> I am Sam </s> <s> Sam I am </s> <s> I and Sam </s> <s> I do not like green eggs and beef </s>"
nltk_tokens = nltk.word_tokenize(word_data)

lip = list(nltk.bigrams(nltk_tokens))
print(lip)


# Python3 code to convert tuple
# into string
def count(listOfTuple):
    flag = False

    # To append Duplicate elements in list
    coll_list = []
    coll_cnt = 0
    for t in listOfTuple:
        # To check if Duplicate exist
        if t in coll_list:
            flag = True
            continue

        else:
            coll_cnt = 0
            for b in listOfTuple:
                if b[0] == t[0] and b[1] == t[1]:
                    coll_cnt = coll_cnt + 1

            # To print count if Duplicate of element exist
            if(coll_cnt > 1):
                print(t, "-", coll_cnt)
            coll_list.append(t)

	if flag == False:
		print("No Duplicates")

# Driver code
print("Test Case 1:")
listOfTuple = [('a', 'e'), ('b', 'x'), ('b', 'x'),
			('a', 'e'), ('b', 'x')]

count(listOfTuple)

print("Test Case 2:")
listOfTuple = [(0, 5), (6, 9), (0, 8)]
count(listOfTuple)



import pandas as pd

# read json into a dataframe
# df_idf=pd.read_json("data/stackoverflow-data-idf.json",lines=True)

# # print schema
# print("Schema:\n\n",df_idf.dtypes)
# print("Number of questions,columns=",df_idf.shape)

df_idf = "<s> I am Sam </s> <s> Sam I am </s> <s> I and Sam </s> <s> I do not like green eggs and beef </s>"



import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))

#show the second 'text' just for fun
df_idf['text'][2]




x = [" I am Sam "," Sam I am" , "I and Sam" ," I do not like green eggs and beef" ]
tfidf = TfidfVectorizer(min_df=2,
                        max_df=50).fit_transform(x)  # model language
tfidf.


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score , precision_score,f1_score,recall_score

lc = SGDClassifier()
X = [(1,2) ,(2,6),(1,3),(3,4)]
y = ["A","B","A","B"]

lc.fit(X[:3], y[:3])

pred_y = lc.predict([(1,2)])
pre_score = precision_score(test_y,pred_y , average="weighted")

#Linear Classifier doesn't have the probability value as it operates differently



from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import  pandas as pd
from sklearn.model_selection import train_test_split # bcz Hold out appraoch

movies_review_path = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/IMDB Dataset.csv"

df  = pd.read_csv(movies_review_path) # reading file
 
reviewDocs_x = df.iloc[(1,2) ,(2,6),(1,3),(3,4)] ## x component

reviewDocs_y = df.iloc["A","B","A","B"]  ## y component

tfidf = TfidfVectorizer(min_df=2,
                        max_df=30).fit_transform(reviewDocs_x)  # model language

train_x,test_x,train_y,test_y = train_test_split(tfidf,
                                                reviewDocs_y,
                                                train_size=0.8
                                                 , 
                                                shuffle=True)  ## spliting data according to hold out cross validation

clf = SGDClassifier().fit(train_x, train_y)  ## put into the Classifer

pred_y = clf.predict(test_x)  # prediction
# pred_y

from sklearn.metrics import accuracy_score , precision_score,f1_score,recall_score
accc = accuracy_score(test_y,pred_y)
pre_score = precision_score(test_y,pred_y , average="weighted")
f1Score = f1_score(test_y,pred_y, average="weighted")
recallScore = recall_score(test_y,pred_y, average="weighted")

print(" Accuracy Score {} \n Precision Score {} \n F1 Score {} \n Recall Score {} ".format(accc,pre_score,f1Score,recallScore))




