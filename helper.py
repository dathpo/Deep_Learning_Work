__author__ = 'Team Alpha'


import sys
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os


class Helper():
    def __init__(self):
        print("TensorFlow version:\t\t%s" % tf.__version__)
        print("Keras version:\t\t\t%s" % keras.__version__)
        print("Python version:\t\t\t%s" % sys.version)
        print()

    def plot_loss_acc(self, epochs, loss, acc, val_loss, val_acc):
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
        filename = 'model'
        graph_path = os.path.join(pwd, '{}'.format(filename + '.pdf'))
        fig.savefig(graph_path, bbox_inches="tight")