#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from helper import Helper
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, InputLayer, Convolution2D, BatchNormalization
from keras.callbacks import CSVLogger
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os


class MNIST(Helper):
    def __init__(self, combination, learning_rate, epochs, batches, seed):
        Helper.__init__(self)
        self.combination = int(combination)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.seed = int(seed)
        self.batch_size = 0
        self.num_classes = 0
        self.input_shape = (0, 0, 0)
        self.run_combination(self.combination)

    def run_combination(self, combination):
        x_train, y_train, x_test, y_test = self.prepare_data()
        if combination == 1:
            model = self.run_first_combo_alt()
        elif combination == 2:
            model = self.run_second_combo()
        else:
            raise Exception("Please input 1 or 2 for the combination to run")
        self.fit_and_evaluate(model, x_train, y_train, x_test, y_test)

    def prepare_data(self):
        tf.set_random_seed(self.seed)
        fashion = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion.load_data()
        """self.batch_size = int(x_train.shape[0] / self.batches)

        print("train set shape:", x_train.shape)
        print("Number of images in train set:", x_train.shape[0])
        print("Number of images in test set:", x_test.shape[0])

        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255"""

        print("Training set shape:", x_train.shape)
        print("Number of images in Training set:", x_train.shape[0])
        print("Number of images in Test set:", x_test.shape[0])

        x_train = np.array(x_train, dtype=np.uint8)
        x_test = np.array(x_test, dtype=np.uint8)
        y_train = np.array(y_train, dtype=np.uint8)
        y_test = np.array(y_test, dtype=np.uint8)

        x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test.reshape(x_test.shape[0], 28, 28, 1)

        if self.combination == 1:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)

        y_train = keras.utils.np_utils.to_categorical(y_train, 10)
        y_test = keras.utils.np_utils.to_categorical(y_test, 10)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test

    def run_first_combo(self, x_train, y_train, x_test, y_test):
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        h = 0

        adam = keras.optimizers.Adam(lr=self.learning_rate)          # default lr=0.001
        sgd = keras.optimizers.SGD(lr=self.learning_rate)            # default lr=0.01
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=adam)
        model.summary()
        return model

    def run_first_combo_alt(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        model = Sequential()
        model.add(InputLayer(input_shape=self.input_shape))
        model.add(BatchNormalization())                                             # Normalisation
        model.add(Convolution2D(64, (4, 4), padding='same', activation='relu'))     # Convolution
        model.add(MaxPooling2D(pool_size=(2, 2)))                                   # Max Pooling
        model.add(Dropout(0.1))                                                     # Dropout
        model.add(Convolution2D(64, (4, 4), activation='relu'))                     # Convolution
        model.add(MaxPooling2D(pool_size=(2, 2)))                                   # Max Pooling
        model.add(Dropout(0.3))                                                     # Dropout
        model.add(Flatten())                                                        # Converting 3D feature to 1D feature vector
        model.add(Dense(256, activation='relu'))                                    # Fully Connected Layer
        model.add(Dropout(0.5))                                                     # Dropout
        model.add(Dense(64, activation='relu'))                                     # Fully Connected Layer
        model.add(BatchNormalization())                                             # Normalization
        model.add(Dense(self.num_classes, activation='softmax'))

        adam = keras.optimizers.Adam(lr=self.learning_rate)  # default lr=0.001

        model.compile(loss='categorical_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])
        model.summary()
        return model

    def run_second_combo(self):
        ## Adapting the fashion MNIST Tutorial
        ## https://www.tensorflow.org/tutorials/keras/basic_classification
        # [ f ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(32, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ g ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(128, activation="relu"),
            #~Dense(128, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ h ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(128, activation="relu"),
            #~Dense(128, activation="relu"),
            #~Dense(128, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ i ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(256, activation="relu"),
            #~Dense(128, activation="relu"),
            #~Dense(64, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ l ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(256, activation="relu"),
            #~Dropout(0.2),
            #~Dense(128, activation="relu"),
            #~Dropout(0.1),
            #~Dense(64, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ m ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(300, activation="relu"),
            #~Dropout(0.3),
            #~Dense(200, activation="relu"),
            #~Dropout(0.2),
            #~Dense(100, activation="relu"),
            #~Dropout(0.1),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)
        # [ n ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(400, activation="relu"),
            #~Dropout(0.25),
            #~Dense(300, activation="relu"),
            #~Dropout(0.2),
            #~Dense(200, activation="relu"),
            #~Dropout(0.1),
            #~Dense(10, activation="softmax")
            #~])
        #~model.compile(
            #~optimizer="adam",
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)

        ## ALTERNATIVELY
        #~model = Sequential()
        #~model.add(Dense(128, input_shape=(784,), activation="relu"))
        #~model.add(Dense(128, input_shape=(784,), activation="relu"))
        #~model.add(10, input_shape=(784,), activation="softmax"))
        #
        #~# For a multi-class classification problem
            #~optimizer="rmsprop",
            #~loss="categorical_crossentropy",
        #~# For a binary classification problem
            #~optimizer="rmsprop",
            #~loss="binary_crossentropy",
        #~# For a mean squared error regression problem
            #~optimizer="rmsprop",
            #~loss="mse"
        return model

    def fit_and_evaluate(self, model, x_train, y_train, x_test, y_test):
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                  write_graph=True, write_images=True)
        tb_callback.set_model(model)
        csv_logger = CSVLogger('training.log', separator=',', append=False)
        """result = model.fit(x_train, y_train,
                           batch_size=self.batches,
                           epochs=self.epochs,
                           verbose=2,
                           validation_split=0.1,
                           callbacks=[tb_callback, csv_logger])

        model.save_weights("fashion_c1.ckpt")
        model.save('fashion_c1.h5')"""
        model.load_weights('fashion_c1.h5')

        # validation_acc = np.amax(result.history['val_acc'])
        # print('Best validation acc of epoch:', validation_acc)
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        train_loss = [0.6487, 0.3869, 0.3327, 0.3049, 0.2841, 0.2653, 0.2532, 0.2433, 0.2323, 0.2249, 0.2152, 0.2115,
                      0.2036,
                      0.1994, 0.1913, 0.1859, 0.1845, 0.1802, 0.1740, 0.1715, 0.1686, 0.1600, 0.1583, 0.1585, 0.1533,
                      0.1524,
                      0.1497, 0.1449, 0.1437, 0.1397, 0.1387, 0.1354, 0.1335, 0.1317, 0.1297, 0.1281, 0.1276, 0.1293,
                      0.1266,
                      0.1206]
        train_acc = [0.7707, 0.8587, 0.8799, 0.8882, 0.8953, 0.9014, 0.9054, 0.9092, 0.9145, 0.9163, 0.9208, 0.9207,
                     0.9248,
                     0.9250, 0.9277, 0.9303, 0.9303, 0.9312, 0.9340, 0.9355, 0.9360, 0.9410, 0.9396, 0.9408, 0.9426,
                     0.9433,
                     0.9429, 0.9454, 0.9473, 0.9479, 0.9479, 0.9483, 0.9491, 0.9504, 0.9512, 0.9517, 0.9523, 0.9512,
                     0.9520,
                     0.9545]
        val_loss = [0.3639, 0.2947, 0.2769, 0.2612, 0.2455, 0.2348, 0.2312, 0.2364, 0.2231, 0.2149, 0.2118, 0.2130,
                    0.2404,
                    0.2056, 0.2107, 0.2267, 0.2065, 0.2029, 0.2107, 0.2072, 0.2044, 0.1980, 0.1997, 0.2057, 0.2188,
                    0.2036,
                    0.2004, 0.2041, 0.2036, 0.2010, 0.2088, 0.2089, 0.2118, 0.2081, 0.2018, 0.2105, 0.2097, 0.2103,
                    0.2057,
                    0.2070]
        val_acc = [0.8665, 0.8880, 0.8970, 0.9022, 0.9072, 0.9118, 0.9138, 0.9155, 0.9187, 0.9205, 0.9220, 0.9218,
                   0.9148,
                   0.9243, 0.9223, 0.9213, 0.9223, 0.9252, 0.9208, 0.9262, 0.9265, 0.9260, 0.9260, 0.9282, 0.9215,
                   0.9278,
                   0.9273, 0.9270, 0.9265, 0.9278, 0.9285, 0.9263, 0.9223, 0.9263, 0.9263, 0.9278, 0.9302, 0.9262,
                   0.9302,
                   0.9293]
        e = range(1, 41)
        return Helper.plot_loss_acc(self, e, train_loss, train_acc, val_loss, val_acc)
        #return self.plot_loss_acc(result.epoch, result.history['loss'], result.history['acc'],
         #                     result.history['val_loss'], result.history['val_acc'])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("epochs", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")
    args = arg_parser.parse_args()
    MNIST(args.combination, args.learning_rate, args.epochs, args.batches, args.seed)
