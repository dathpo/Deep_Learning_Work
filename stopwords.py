#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:06:09 2019

@author: david
"""
import re
import nltk
from nltk.corpus import stopwords

stopwords_eng = set(stopwords.words("english"))
stopwords_eng.remove('no')
stopwords_eng.remove('not')
stopwords_eng.remove("shouldn't")
stopwords_eng.remove("wasn't")
stopwords_eng.remove("weren't")
stopwords_eng.remove("wouldn't")
stopwords_eng.remove("won't")
stopwords_eng.remove("but")
stopwords_eng.remove("aren't")
stopwords_eng.remove("aren'")
stopwords_eng.remove("couldn'")
stopwords_eng.remove("couldn't")
stopwords_eng.remove("didn'")
stopwords_eng.remove("didn't")
stopwords_eng.remove("very")



imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index

word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def stopword_removal(data):
    processed_reviews = []
    for review in data:
        word_review = ' '.join([reverse_word_index.get(i, '?') for i in review])
        words = [word for word in word_review.split() if word not in stopwords_eng]
        indexed_words = []
        for w in words:
            index = word_index.get(w)
            indexed_words.append(index)
        processed_reviews.append(indexed_words)
    print(processed_reviews[0])
    print(train_data[0])
    
train_data = stopword_removal(train_data)
