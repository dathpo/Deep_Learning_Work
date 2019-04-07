#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from packageinfo import PackageInfo
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation, InputLayer, Convolution2D, BatchNormalization
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
        self.num_classes = 0
        self.input_shape = (0, 0, 0)
        self.run_combination(int(combination))

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

        result = model.fit(x_train, y_train,
                           batch_size=self.batches,
                           epochs=self.epochs,
                           verbose=2,
                           validation_split=0.1,
                           callbacks=[tb_callback])

        model.save_weights("fashion_c1.ckpt")
        model.save('fashion_c1.h5')

        validation_acc = np.amax(result.history['val_acc'])
        print('Best validation acc of epoch:', validation_acc)
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("epochs", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")
    args = arg_parser.parse_args()
    MNIST(args.combination, args.learning_rate, args.epochs, args.batches, args.seed)
