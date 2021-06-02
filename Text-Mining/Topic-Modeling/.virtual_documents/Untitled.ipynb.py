## import lib
import  pandas as pd
import numpy as np

import gensim
from gensim.utils import simple_preprocess as sp
from gensim.parsing.preprocessing import STOPWORDS as stw
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import nltk
nltk.download("wordnet")






publishedBookPath = "/home/iffishells/Desktop/Data-Science/Text-Mining/Datasets/abcnews-date-text.csv"
data = pd.read_csv(publishedBookPath)

## sample data
data = data.head(10)
# data








data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text


## pre-processing
# --------------------------------------------------

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos="v"))

def preprocess(text):
    
    results = []
    for token in sp(text):
        if token not in stw and len(token) > 3:
            results.append(lemmatize_stemming(token))
    return results




## selecting document after pre-processing 
doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))






