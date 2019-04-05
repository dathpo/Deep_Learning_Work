__author__ = 'Team Alpha'


import sys
import tensorflow as tf


class PackageInfo():
    def __init__(self):
        print("Python version:", sys.version)
        print("Tensorflow version:", tf.__version__)
        print()