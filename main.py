__author__ = 'Team Alpha'

from imdb import IMDb
from fashion import Fashion


def main(dataset):
    if dataset == 0:
        Fashion(
            combination=2,
            learning_rate=0.001,
            epochs=5,
            batches=128,
            seed=12345
        )
        
    elif dataset == 1:
        IMDb(
            combination=1,
            learning_rate=0.001,
            epochs=2,
            batches=512,
            seed=12345
        )


if __name__ == "__main__":
    main(0)
