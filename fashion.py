#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Team Alpha'

from helper import Helper, arg_parser
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
import numpy as np
import random as rand


class Fashion(Helper):
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
            model = self.run_first_combo()
            modelname = "fashion_1_" + str(self.learning_rate) + "_" + str(self.epochs) + "_" + str(self.batches) + "_" + str(self.seed) + ""
        elif combination == 2:
            model = self.run_second_combo()
            modelname = "fashion_2_" + str(self.learning_rate) + "_" + str(self.epochs) + "_" + str(self.batches) + "_" + str(self.seed) + ""
        else:
            raise Exception("Please input 1 or 2 for the combination to run")
        # modelname = "fashion-model"
        data = x_train, y_train, x_test, y_test
        result = Helper.fit_and_evaluate(self, model, data, self.batches, self.epochs, modelname)
        Helper.plot_loss_acc(self, result.epoch, result.history['loss'], result.history['acc'],
                             result.history['val_loss'], result.history['val_acc'], modelname)

    def prepare_data(self):
        config = tf.ConfigProto(inter_op_parallelism_threads=1)
        session = tf.Session(config=config)
        K.set_session(session)
        rand.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        fashion = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion.load_data()

        print("Training set shape:", x_train.shape)
        print("Number of images in Training set:", x_train.shape[0])
        print("Number of images in Test set:", x_test.shape[0])

        x_train = np.array(x_train, dtype=np.uint8)
        x_test = np.array(x_test, dtype=np.uint8)
        y_train = np.array(y_train, dtype=np.uint8)
        y_test = np.array(y_test, dtype=np.uint8)

        if self.combination == 1:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
            y_train = keras.utils.np_utils.to_categorical(y_train, 10)
            y_test = keras.utils.np_utils.to_categorical(y_test, 10)

        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        return x_train, y_train, x_test, y_test

    def run_first_combo(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        model = Sequential()
        model.add(BatchNormalization(input_shape=self.input_shape))                 # Normalisation
        model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))            # Convolution
        model.add(MaxPooling2D(pool_size=(2, 2)))                                   # Max Pooling
        model.add(Dropout(0.1))                                                     # Dropout
        model.add(Conv2D(64, (4, 4), activation='relu'))                            # Convolution
        model.add(MaxPooling2D(pool_size=(2, 2)))                                   # Max Pooling
        model.add(Dropout(0.3))                                                     # Dropout
        model.add(Flatten())                                                        # Converting 3D feat. to 1D feat.
        model.add(Dense(256, activation='relu'))                                    # Fully Connected Layer
        model.add(Dropout(0.5))                                                     # Dropout
        model.add(Dense(64, activation='relu'))                                     # Fully Connected Layer
        model.add(BatchNormalization())                                             # Normalization
        model.add(Dense(self.num_classes, activation='softmax'))

        """model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
        model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1)))
        model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
        model.add(Activation(activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(256, activation='linear'))
        model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='linear'))
        model.add(BatchNormalization(axis=1, epsilon=1e-05, momentum=0.9))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation="linear"))"""

        adam = keras.optimizers.Adam(lr=self.learning_rate)                         # default lr=0.001
        sgd = keras.optimizers.SGD(lr=self.learning_rate)                           # default lr=0.01

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        model.summary()
        return model

    def run_second_combo(self):
        adam = keras.optimizers.Adam(lr=self.learning_rate)  # default lr=0.001
        sgd = keras.optimizers.SGD(lr=self.learning_rate)  # default lr=0.01
        ## Configurations
        # [ f (BEST) ]
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(500, activation="relu"),
            Dropout(0.20),
            Dense(400, activation="relu"),
            Dropout(0.25),
            Dense(300, activation="relu"),
            Dropout(0.30),
            Dense(200, activation="relu"),
            Dropout(0.35),
            Dense(10, activation="softmax")
            ])
        # [ g ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(256, activation="relu"),
            #~Dropout(0.2),
            #~Dense(128, activation="relu"),
            #~Dropout(0.1),
            #~Dense(64, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
       # [ h ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(128, activation="relu"),
            #~Dense(128, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])

        # Optimizer for the three best models
        model.compile(
            optimizer=adam,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
            )

        # [ i ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(64, activation="relu"),
            #~Dense(10, activation="softmax")
            #~])
        # [ j (WORST ]
        #~model = Sequential([
            #~Flatten(input_shape=(28, 28)),
            #~Dense(32, activation="sigmoid"),
            #~Dense(10, activation="softmax")
            #~])

        # Optimizer for the two worst models
        #~model.compile(
            #~optimizer=sgd,
            #~loss="sparse_categorical_crossentropy",
            #~metrics=["accuracy"]
            #~)

        model.summary()
        return model


if __name__ == "__main__":
    args = arg_parser()
    Fashion(args.combination, args.learning_rate, args.epochs, args.batches, args.seed)
