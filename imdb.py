from __future__ import absolute_import, division, print_function

__author__ = 'Team Alpha'

import sys
import tensorflow as tf
from tensorflow import keras
from packageinfo import PackageInfo
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D
from keras.layers import Convolution1D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import initializers, regularizers, constraints, optimizers, layers
import re
import nltk
from nltk.corpus import stopwords

class IMDb(PackageInfo):
    vocab_size = 10000
    def __init__(self, combination, learning_rate, epochs, batches, seed):
        PackageInfo.__init__(self)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.seed = seed
        self.select_combination(combination)

    def select_combination(self, combination):
        tf.set_random_seed(self.seed)
        train_data, train_labels, test_data, test_labels = self.prepare_data()
        if combination == 1:
            self.run_first_combo(train_data, train_labels, test_data, test_labels)
        elif combination == 2:
            self.run_second_combo(train_data, train_labels, test_data, test_labels)
        else:
            print("Please input 1 or 2 for the combination to run")

        
    def prepare_data(self):
        
        stopwords_eng = set(stopwords.words("english"))
        stopwords_eng.remove('no')
        stopwords_eng.remove('not')
        stopwords_eng.remove("shouldn't")
        stopwords_eng.remove("wasn't")
        stopwords_eng.remove("weren't")
        stopwords_eng.remove("wouldn't")
        stopwords_eng.remove("won't")
        
        imdb = keras.datasets.imdb
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=self.vocab_size)
    
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
                
            return(processed_reviews)
    
        train_data = stopword_removal(train_data)
        
        test_data = stopword_removal(test_data)
                    
    
        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index['<PAD>'],
                                                            padding='post',
                                                            maxlen=256)

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
        
        return train_data, train_labels, test_data, test_labels
    
    def run_first_combo(self, train_data, train_labels, test_data, test_labels):
        """
        First combination with LSTM RNN here
        """
        c1_model = self.build_c1_model(train_data, train_labels)
        self.test_c1_model(c1_model, test_data, test_labels)    
        
    def build_c1_model(self, train_data, train_labels):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 16))
        model.add(Bidirectional(LSTM(32, return_sequences = True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation=tf.nn.relu))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation=tf.nn.sigmoid))

        model.summary()

        # Loss function and optimizer

        # Binary cross-entropy loss function suited to binary classification
        
        opt = SGD(lr=self.learning_rate)
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        # During training, want to check accuracy of model on data it's not seen before. Validation.

        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]   
        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=self.epochs,
                            batch_size=self.batches,
                            validation_data=(x_val, y_val),
                            verbose=1)
        return model
            
    def test_c1_model(self, model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(results)

    def run_second_combo(self, train_data, train_labels, test_data, test_labels):
        c2_model = self.build_c2_model(train_data, train_labels)
        self.test_c2_model(c2_model, test_data, test_labels)
        
    def build_c2_model(self, train_data, train_labels):
        
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        model.summary()

        # Loss function and optimizer

        # Binary cross-entropy loss function suited to binary classification

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        # During training, want to check accuracy of model on data it's not seen before. Validation.

        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=self.epochs,
                            batch_size=self.batches,
                            validation_data=(x_val, y_val),
                            verbose=1)
        
        return model

    def test_c2_model(self, model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(results)