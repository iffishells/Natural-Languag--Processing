# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:08:44 2019

@author: Taimoor
"""

corpus = '<html><head></head><body><h1>Paragraph Heading</h1><p>This is some text. <a href="">The original price was $500 but now only USD250 </a> This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is <em>some text.</em> <strong>This is some text.</strong> This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. This is some text. </p></body></html>'
print('Raw data: ', corpus)
import re
tags = re.compile(r'<.*?>')
corpus = tags.sub('', corpus)

prices = re.compile(r'(USD|\$)[0-9]+')
corpus = prices.sub('', corpus)

print('normalized data: ', corpus)