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
        self.combination = combination
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batches = batches
        self.seed = seed
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
        print('x_train shape:', x_train.shape)
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

        """
        update_interval = int(epochs / 5)
        batch_size = int(len(mnist.train.labels) / batches)
        # Extracting the dimensions of the problem' space.
        m, n = x_train.data.shape

        # Populating the TensorFlow environment
        # Creating the constants
        x = tf.constant(    x_train,
                            dtype = tf.float32,
                            name = "x"
                            )
        y = tf.constant(    x_train,
                            dtype = tf.float32,
                            name = "y"
                            )

        ## Creating the variables
        theta = tf.Variable(    tf.random_uniform(  [n, 1],
                                                    -1.0,
                                                    1.0
                                                    ),
                                name="theta"
                                )

        ## Creating the optimizer
        ## calculated using tfs gradient descent optimizer
        #~optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate);

        ## calculated using tfs momentum descent optimizer - faster
        optimizer = tf.train.MomentumOptimizer( learning_rate = learning_rate,
                                                momentum = 0.9
                                                )

        ## Creating the operations
        y_pred = tf.matmul(
            x,
            theta,
            name="predictions"
        )
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error),
                             name="mse"
                             )
        training_op = optimizer.minimize(mse)

        ## Initializing the TensorFlow environment
        init = tf.global_variables_initializer()

        ## Running the operations between the variables using a TensorFlow.Session.
        ## Class Session: returns TensorFlow operations
        ## A Session object encapsulates the environment in which Operation objects
        ## are executed, and Tensor objects are evaluated.
        ## Note: using the context manager "with", it is not necessary to release
        ## the sessions resources when no longer required (using tf.Session.close).
        with tf.Session() as sess:
            ## Completing variables and graph structure initialization.
            sess.run(init)
            ## For the requested amount of times (epochs = iterations) train the ANN.
            for epoch in range(epochs):
                avg_cost = 0
                for batch in range(batch_size):
                    batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
                    _, cost = sess.run([optimizer, cross_entropy],
                                    feed_dict = {x: batch_x, y: batch_y})
                    avg_cost += cost / batches
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

                ## At regular interval, including iteration 0, print a status update
                ## on the achieved effectiveness of the training (the value of the
                ## cost function).
                if epoch % update_interval == 0:
                    print(  "Epoch ",
                            epoch,
                            " MSE = ",
                            mse.eval()
                            )
                    ## Compute the output of the graph (run the operations); i.e.
                    ## train the ANN.
                    sess.run(training_op)
            ## Compute and print the final value of the wheights (is thatwhat theta
            ## is?) obtained at the end of the training session (the epochs).
            best_theta = theta.eval()
            print(best_theta)

            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
            """

    def run_second_combo(self, mnist):
        return 0


# [ References ]
# [1] Python TensorFlow Tutorial – Build a Neural Network
#   - https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
