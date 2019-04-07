#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from packageinfo import PackageInfo
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
import numpy as np
import argparse


class MNIST(PackageInfo):
    def __init__(self, combination, learning_rate, epochs, batches, seed):
        PackageInfo.__init__(self)
        self.combination = int(combination)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.seed = int(seed)
        self.batch_size = 0
        self.select_combination(int(combination))

    def select_combination(self, combination):
        x_train, y_train, x_test, y_test = self.prepare_data()
        if combination == 1:
            self.run_first_combo(x_train, y_train, x_test, y_test)
        elif combination == 2:
            self.run_second_combo(x_train, y_train, x_test, y_test)
        else:
            raise Exception("Please input 1 or 2 for the combination to run")

    def prepare_data(self):
        tf.set_random_seed(self.seed)
        fashion = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion.load_data()
        self.batch_size = int(x_train.shape[0] / self.batches)

        print("train set shape:", x_train.shape)
        print("Number of images in train set:", x_train.shape[0])
        print("Number of images in test set:", x_test.shape[0])

        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        return x_train, y_train, x_test, y_test

    def run_first_combo(self, x_train, y_train, x_test, y_test):
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.summary()

        h = 0

        adam = keras.optimizers.Adam(lr=self.learning_rate)          # default lr=0.001
        sgd = keras.optimizers.SGD(lr=self.learning_rate)            # default lr=0.01

        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                 write_graph=True, write_images=True)
        tbCallBack.set_model(model)

        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer=sgd)

        result = model.fit(x_train, y_train,
                           batch_size=self.batches,
                           epochs=self.epochs,
                           verbose=2,
                           validation_split=0.1,
                           callbacks=[tbCallBack])

        model.save_weights("fashion_c1.ckpt")
        model.save('fashion_c1.h5')

        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def run_second_combo(self, x_train, y_train, x_test, y_test):
        ## Adapting the fashion MNIST Tutorial
        ## https://www.tensorflow.org/tutorials/keras/basic_classification

        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax")
            ])
        # alternatively
        #~model = Sequential()
        #~model.add(Dense(128, input_shape=(784,), activation="relu"))
        #~model.add(Dense(128, input_shape=(784,), activation="relu"))
        #~model.add(10, input_shape=(784,), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
            )
        ## ALTERNATIVELY
        #~# For a multi-class classification problem
            #~optimizer="rmsprop",
            #~loss="categorical_crossentropy",
        #~# For a binary classification problem
            #~optimizer="rmsprop",
            #~loss="binary_crossentropy",
        #~# For a mean squared error regression problem
            #~optimizer="rmsprop",
            #~loss="mse"

        print("bat", self.batch_size)

        model.fit(
            x_train,
            y_train,
            epochs = self.epochs,
            batch_size = self.batch_size
            )

        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)

        return 0

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("epochs", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")
    args = arg_parser.parse_args()
    MNIST(args.combination, args.learning_rate, args.epochs, args.batches, args.seed)
