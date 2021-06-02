# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 02:41:30 2019

@author: Taimoor
"""

corpus = 'He owes me 22.50 dollars. Which is due by the next day.'
print(corpus.split('.'))

from nltk.tokenize import sent_tokenize, word_tokenize
print(sent_tokenize(corpus))

for sent in sent_tokenize(corpus):
    print(word_tokenize(sent))