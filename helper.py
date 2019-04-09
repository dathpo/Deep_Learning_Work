__author__ = 'Team Alpha'


import sys
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
from keras.callbacks import CSVLogger
from keras.models import load_model
import numpy as np
import argparse
from signal import SIGINT, signal
import seaborn as sns; sns.set()
import nltk


class Helper:
    def __init__(self):
        print("TensorFlow version:\t\t%s" % tf.__version__)
        print("Keras version:\t\t\t%s" % keras.__version__)
        print("NLTK version:\t\t\t%s" % nltk.__version__)
        print("Python version:\t\t\t%s" % sys.version)
        print()

    def fit_and_evaluate(self, model, data, batches, epochs, filename):
        x_train, y_train, x_test, y_test = data
        pwd = os.path.abspath(os.path.dirname(__file__))
        tb_callback = keras.callbacks.TensorBoard(log_dir=pwd + "/logs/", histogram_freq=0,
                                                  write_graph=True, write_images=True)
        tb_callback.set_model(model)
        csv_logger = CSVLogger(filename=pwd + "/logs/" + filename + ".log", separator=',', append=False)
        result = model.fit(x_train, y_train,
                           batch_size=batches,
                           epochs=epochs,
                           verbose=1,
                           validation_split=0.1,
                           callbacks=[tb_callback, csv_logger])

        model.save_weights(pwd + "/" + filename + ".ckpt")
        # model.save(filename + ".h5")
        # model.load_weights(filename + ".ckpt")                     # .h5 also works
        # model = load_model(filename + ".h5")

        validation_acc = np.amax(result.history['val_acc'])
        print('\nBest Validation Accuracy:', validation_acc)
        print()
        train_loss = result.history['loss'][-1]
        train_acc = result.history['acc'][-1]
        print("Train Loss:", train_loss)
        print("Train Accuracy:", train_acc)
        print()
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print("\nTest Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        return result

    def plot_loss_acc(self, epochs, loss, acc, val_loss, val_acc, filename):
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
        ax_loss.plot(epochs, loss, label="Train Loss", c="blue")
        ax_loss.plot(epochs, val_loss, label="Validation Loss", c="red")
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('epoch')
        ax_loss.legend(loc='best')
        ax_acc.plot(epochs, acc, label="Train Accuracy", c="blue")
        ax_acc.plot(epochs, val_acc, label="Validation Accuracy", c="red")
        ax_acc.set_title('Accuracy')
        ax_acc.set_xlabel('epoch')
        ax_acc.legend(loc='best')
        pwd = os.path.abspath(os.path.dirname(__file__))
        graph_path = os.path.join(pwd, '{}'.format(filename + '.pdf'))
        fig.savefig(graph_path, bbox_inches="tight")


def arg_parser():
    signal(SIGINT, SIGINT_handler)
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("epochs", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")
    args = arg_parser.parse_args()
    check_arguments(args)
    return args


def check_arguments(arguments):
    """
    Check the validity of all arguments and exit if any is invalid
    """
    quit = False
    for argument, value in vars(arguments).items():
        try:
            float(value)
        except:
            print("{} must be numeric".format(argument))
            quit = True
    if quit:
        exit(1)


def SIGINT_handler(signal, frame):
    """
    ISR to handle the Ctrl-C combination and stop the program in a clean way
    """
    exit(2)
