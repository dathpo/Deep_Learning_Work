__author__ = 'Team Alpha'


import sys
import tensorflow as tf
import keras


class PackageInfo():
    def __init__(self):
        print("TensorFlow version:\t\t%s" % tf.__version__)
        print("Keras version:\t\t\t%s" % keras.__version__)
        print("Python version:\t\t\t%s" % sys.version)
        print()