from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
import numpy as np


__author__ = 'Team Alpha'


class IMDb:
    def __init__(
            self,
            combination,
            learning_rate,
            epochs,
            batches,
            seed
    ):
        self.combination = combination
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.seed = seed
        self.select_combination(self.combination)

    def select_combination(self, combination):
        if combination == 1:
            self.run_first_combo()
        elif combination == 2:
            self.run_second_combo()
        else:
            print("Please input 1 or 2 for the combination to run")

    def run_first_combo(self):
        """
        First combination with LSTM RNN here
        """
        imdb = keras.datasets.imdb
        return 0

    def run_second_combo(self):
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

        def decode_review(text):
            return ' '.join([reverse_word_index.get(i, '?') for i in text])

        decode_review(train_data[0])

        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                                value=word_index['<PAD>'],
                                                                padding='post',
                                                                maxlen=256)

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                               value=word_index["<PAD>"],
                                                               padding='post',
                                                               maxlen=256)

        vocab_size = 10000

        # Layers will be stacked sequentially
        model = keras.Sequential()

        # First layer gets integer-encoded vocab and gets embedding vector for each word-index
        model.add(keras.layers.Embedding(vocab_size, 16))
        #
        # # Pooling layer returns a fixed-length output vector for each example by averaging over sequence dimension.
        # # model.add(GlobalMaxPool1D())
        model.add(keras.layers.GlobalAveragePooling1D())
        # model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        #
        # model.add(keras.layers.Dropout(0.5))

        # Fixed length output vector is piped through a fully-connected (dense) layer with 16 nodes
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        # model.add(Dense(16, activation=tf.nn.relu))
        # model.add(keras.layers.Dropout(0.5))
        # Last dense layer connected with a single output node with sigmoid activation function giving float between 0 and 1
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

        results = model.evaluate(test_data, test_labels)

        print(results)

        # import re
        # from nltk.stem import WordNetLemmatizer
        # from ntlk.corpus import stopwords
        #
        # stop_words = set(stopwords.words('english'))
        # lemmatizer = WordNetLemmatizer()
        #
        # def clean_