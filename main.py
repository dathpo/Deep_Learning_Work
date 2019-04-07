__author__ = 'Team Alpha'

from imdb import IMDb
from mnist import MNIST, prepare_data, run_first_combo
from hyperas import optim
from hyperopt import Trials, tpe
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation


def main(dataset):
    """if dataset == 0:
        mnist_model = MNIST(
            combination=1,
            learning_rate=0.05,
            epochs=50,
            batches=75,
            seed=12345
        )
    elif dataset == 1:
        imdb_model = IMDb(
            combination=1,
            learning_rate=0.001,
            epochs=40,
            batches=500,
            seed=12345
        )
        """

    best_run, best_model = optim.minimize(model=run_first_combo,
                                          data=prepare_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    x_train, y_train, x_test, y_test = prepare_data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)


if __name__ == "__main__":
    main(0)