from __future__ import absolute_import, division, print_function

__author__ = 'Team Alpha'

import tensorflow as tf
from tensorflow import keras
from packageinfo import PackageInfo
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import MaxPooling1D
from keras.layers import Conv1D
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.corpus import stopwords

class IMDb(PackageInfo):
    vocab_size = 20000
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
    
        clean_train_data = stopword_removal(train_data)
        clean_test_data = stopword_removal(test_data)
                    
        train_data = keras.preprocessing.sequence.pad_sequences(clean_train_data,
                                                            value=word_index['<PAD>'],
                                                            padding='post',
                                                            maxlen=512)

        test_data = keras.preprocessing.sequence.pad_sequences(clean_test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=512)
        
        return train_data, train_labels, test_data, test_labels
    
    def run_first_combo(self, train_data, train_labels, test_data, test_labels):
        c1_model = self.build_c1_model(train_data, train_labels)
        self.test_c1_model(c1_model, test_data, test_labels)    
        
    def build_c1_model(self, train_data, train_labels):
        model = Sequential()
        
#        model.add(Embedding(self.vocab_size, 16))
#        model.add(Bidirectional(LSTM(32, return_sequences = True)))
#        model.add(GlobalMaxPool1D())
#        model.add(Dense(20, activation=tf.nn.relu))
#        model.add(Dropout(0.05))
#        model.add(Dense(1, activation=tf.nn.sigmoid))
        
        model.add(Embedding(self.vocab_size, 65, input_length=512))
        model.add(Conv1D(filters=65, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=512))
        model.add(Flatten())
        model.add(Dense(400, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        opt = SGD(lr=self.learning_rate)
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        # During training, want to check accuracy of model on data it's not seen before. Validation.

        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]   
        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]
        
        tbcallback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                            write_graph=True, write_images=True)
        
        performance = model.fit(partial_x_train,
                  partial_y_train,
                  epochs=self.epochs,
                  batch_size=self.batches,
                  validation_data=(x_val, y_val),
                  verbose=1,
                  callbacks=[tbcallback])
        
        tbcallback.set_model(model)
        
        import matplotlib.pyplot as plt
        history_dict = performance.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()       
        plt.show()
        
        plt.clf()   
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()      
        plt.show()
        return model
            
    def test_c1_model(self, model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(results)

    def run_second_combo(self, train_data, train_labels, test_data, test_labels):
        c2_model = self.build_c2_model(train_data, train_labels)
        self.test_c2_model(c2_model, test_data, test_labels)
        
    def build_c2_model(self, train_data, train_labels):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 32, input_length=256))
        model.add(Flatten())
        model.add(Dense(250, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        model.summary()
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        
        # During training, want to check accuracy of model on data it's not seen before. Validation.

        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]

        performance = model.fit(partial_x_train,
                  partial_y_train,
                  epochs=self.epochs,
                  batch_size=self.batches,
                  validation_data=(x_val, y_val),
                  verbose=1)
        
        import matplotlib.pyplot as plt
        history_dict = performance.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()       
        plt.show()
        
        plt.clf()   
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()      
        plt.show()
        return model

    def test_c2_model(self, model, test_data, test_labels):
        results = model.evaluate(test_data, test_labels)
        print(results)