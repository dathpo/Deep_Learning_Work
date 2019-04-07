#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from packageinfo import PackageInfo
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D



class MNIST(PackageInfo):
    def __init__(self, combination, learning_rate, epochs, batches, seed):
        PackageInfo.__init__(self)
        self.combination = int(combination)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batches = int(batches)
        self.seed = int(seed)
        self.prepare_data()

    def prepare_data(self):
        tf.set_random_seed(self.seed)
        mnist = tf.keras.datasets.mnist
        if self.combination == 1:
            self.run_first_combo(mnist)
        elif self.combination == 2:
            self.run_second_combo(mnist)
        else:
            print("Please input 1 or 2 for the combination to run")

    def run_first_combo(self, mnist):

        # Implemented following [1]

        # !!! NEED TO HANDLE GRACEFULLY WHEN NO DATASET IS PASSED TO THE FUNCTION.

        # Optimization varibales
        # epochs = 1000
        # learning_rate = 0.01

        # Function's environment variables
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255

        print('Number of images in x_train', x_train.shape[0])
        print('Number of images in x_test', x_test.shape[0])
        print("Shape:", x_train.shape)

        """model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])"""

        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        print(model.evaluate(x_test, y_test))

    def run_second_combo(self, mnist):
        ## Adapting the fashion MNIST Tutorial
        ## https://www.tensorflow.org/tutorials/keras/basic_classification

        tf.set_random_seed(self.seed)
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
        batch_size = int(train_x.shape[0] / self.batches)

        # Normalizing the RGB codes by dividing it to the max RGB value.
        train_x = train_x.astype('float32') / 255
        test_x = test_x.astype('float32') / 255

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
        # [ l ]
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(300, activation="relu"),
            Dropout(0.3),
            Dense(200, activation="relu"),
            Dropout(0.2),
            Dense(100, activation="relu"),
            Dropout(0.1),
            Dense(10, activation="softmax")
            ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
            )

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

        model.fit(
            train_x,
            train_y,
            epochs = self.epochs,
            batch_size = batch_size
            )

        test_loss, test_accuracy = model.evaluate(test_x, test_y)

        print("Test Accuracy:", test_accuracy)
        print("Test Loss:", test_loss)

        return 0
