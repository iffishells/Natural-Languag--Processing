# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:44:56 2019

@author: Taimoor
"""

corpus = '    This is my duMmy Dataset. It is part of taught course on text mining.   '

#1. to lowercase
corpus = corpus.lower()

#2 remove white spaces
corpus = corpus.strip()

#3 remove punctuations
from string import punctuation as punc
for ch in corpus:
    if ch in punc:
        corpus = corpus.replace(ch, '')
        
print(corpus)        
