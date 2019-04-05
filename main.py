__author__ = 'Team Alpha'

from imdb import IMDb
from mnist import MNIST

def main():
    """
    """
    MNIST(
        combination=1,
        learning_rate=0.05,
        epochs=50,
        batches=75,
        seed=12345
    )

    """
    IMDb(
        combination=2,
        learning_rate=0.05,
        epochs=60,
        batches=700,
        seed=12345
    )
    """


if __name__ == "__main__":
    main()