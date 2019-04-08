__author__ = 'Team Alpha'

from imdb import IMDb
from mnist import MNIST


def main(dataset):
    if dataset == 0:
        MNIST(
            combination=1,
            learning_rate=0.001,
            epochs=5,
            batches=32,
            seed=12345
        )
    elif dataset == 1:
        IMDb(
            combination=2,
            learning_rate=0.04,
            epochs=5,
            batches=50,
            seed=12345
        )


if __name__ == "__main__":
    main(1)