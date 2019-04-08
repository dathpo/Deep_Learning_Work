from __future__ import absolute_import, division, print_function

__author__ = 'Team Alpha'

import tensorflow as tf
from tensorflow import keras
from helper import Helper, arg_parser
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.corpus import stopwords


class IMDb(Helper):
    vocab_size = 5000

    def __init__(self, combination, learning_rate, epochs, batches, seed):
        Helper.__init__(self)
        self.combination = int(combination)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.seed = int(seed)
        self.run_combination(self.combination)

    def run_combination(self, combination):
        train_data, train_labels, test_data, test_labels = self.prepare_data()
        if combination == 1:
            model = self.run_first_combo()
            modelname = "imdb_c1_" + str(self.learning_rate) + "_" + str(self.epochs) + "_" + str(self.batches) + "_" + str(self.seed) + ""
        elif combination == 2:
            model = self.run_second_combo()
            modelname = "imdb_c2_" + str(self.learning_rate) + "_" + str(self.epochs) + "_" + str(self.batches) + "_" + str(self.seed) + ""
        else:
            raise Exception("Please input 1 or 2 for the combination to run")
        data = train_data, train_labels, test_data, test_labels
        result = Helper.fit_and_evaluate(self, model, data, self.batches, self.epochs, modelname)
        Helper.plot_loss_acc(self, result.epoch, result.history['loss'], result.history['acc'],
                             result.history['val_loss'], result.history['val_acc'], modelname)
        
    def prepare_data(self):
        tf.set_random_seed(self.seed)
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
        stopwords_eng.remove("aren")
        stopwords_eng.remove("couldn")
        stopwords_eng.remove("couldn't")
        stopwords_eng.remove("didn")
        stopwords_eng.remove("didn't")
        stopwords_eng.remove("very")

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
    
        clean_train_data = stopword_removal(train_data)
        clean_test_data = stopword_removal(test_data)
                    
        train_data = keras.preprocessing.sequence.pad_sequences(clean_train_data,
                                                            maxlen=512)

        test_data = keras.preprocessing.sequence.pad_sequences(clean_test_data,
                                                           maxlen=512)
        
        return train_data, train_labels, test_data, test_labels
        
    def run_first_combo(self):
        model = Sequential()

        model.add(Embedding(self.vocab_size, 250, input_length=512))
        model.add(Conv1D(filters=250, kernel_size=3, padding='same', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(400, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.summary()
        return model
        
    def run_second_combo(self):
        
        model = Sequential()
        model.add(Embedding(self.vocab_size, 150, input_length=512))
        model.add(Flatten())
        model.add(Dense(250, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        model.summary()
     
#        model.add(Embedding(self.vocab_size, 128))
#        model.add(LSTM(128, dropout=0.02, recurrent_dropout=0.02))
##        model.add(MaxPooling1D(pool_size=512))
##        model.add(Flatten())
##        model.add(Dense(16, activation=tf.nn.relu))
#        model.add(Dense(1, activation=tf.nn.sigmoid))

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        model.summary()
        return model


if __name__ == "__main__":
    args = arg_parser()
    IMDb(args.combination, args.learning_rate, args.epochs, args.batches, args.seed)
