#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from packageinfo import PackageInfo
from hyperas.distributions import choice, uniform
from hyperopt import STATUS_OK
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
import numpy as np


class MNIST(PackageInfo):
    def __init__(self, combination, learning_rate, epochs, batches, seed):
        PackageInfo.__init__(self)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.seed = seed
        self.select_combination(combination)

    def select_combination(self, combination):
        x_train, y_train, x_test, y_test = prepare_data()
        if combination == 1:
            run_first_combo(x_train, y_train, x_test, y_test)
        elif combination == 2:
            run_second_combo(x_train, y_train, x_test, y_test)
        else:
            raise Exception("Please input 1 or 2 for the combination to run")

    def old_function(self):
        pass


def run_second_combo(x_train, y_train, x_test, y_test):
    return 0


def prepare_data():
    tf.set_random_seed(12345)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print()
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    print("Shape:", x_train.shape)
    print()
    return x_train, y_train, x_test, y_test


def run_first_combo(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
    model.add(Dense({{choice([128, 256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr={{uniform(0, 1)}})  # self.learning_rate
    # opt = keras.optimizers.SGD(lr={{uniform(0, 1)}})
    # opt = keras.optimizers.RMSprop(lr={{uniform(0, 1)}})
    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    result = model.fit(x_train, y_train,
                       batch_size={{choice([64, 128, 256])}},  # self.batches
                       epochs={{choice([2, 5, 10])}},  # self.epochs
                       verbose=2,
                       validation_split=0.1)

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
