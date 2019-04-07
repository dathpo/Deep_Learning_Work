#!/usr/bin/env python
# -*- coding: utf-8 -*-
# mnist.py
"""
"""
__author__ = "Team Alpha"


#import os
from sys import exit;
from argparse import ArgumentParser;

from mnist import MNIST


def SIGINT_handler( signal,
                    frame
                    ):
    # ISR to handle the Ctrl-C combination and stop the program in a clean way.
    exit(2);


def main(   combination = 1,
            learning_rate = 0.001,
            epochs = 5,
            batches = 32,
            seed = 12345
            ):
    """
    @param combination: Indicates which ANN architecture to run.
    @type combination: Integer greater or equal than 1.
    @param learning_rate: Learning Rate parameter.
    @type learning_rate: Positive real number
    @param epochs: Number of iterations used to train the ANN.
    @type epochs: Positive integer.
    @param batches: Number of batches to use during training.
    @type batches: Integer greater or equal than 1.
    @param seed: Number to initialize the network's random generator.
    @type seed: Integer.
    @return: TBD
    """
    MNIST(
        combination = combination,
        learning_rate = learning_rate,
        epochs = epochs,
        batches = batches,
        seed = seed
        )

def check_arguments(arguments):
# Check the validity of all arguments and exit if any is invalid
    quit = False;
    for argument, value in vars( arguments ).items() :
        try:
            float(value);
        except:
            print( "{} must be numeric".format( argument ) );
            quit = True;
    if quit : exit(1);


if __name__ == '__main__':
    from signal import SIGINT, signal, pause
    signal( SIGINT,
            SIGINT_handler
            )

    arg_parser = ArgumentParser( description = "Assignment Program" );
    arg_parser.add_argument(    "combination",
                                help="Flag to indicate which network to run"
                                );
    arg_parser.add_argument(    "learning_rate",
                                help="Learning Rate parameter"
                                );
    arg_parser.add_argument(    "epochs",
                                help="Number of epochs (iterations) to perform"
                                );
    arg_parser.add_argument(    "batches",
                                help="Number of batches to use"
                                );
    arg_parser.add_argument(    "seed",
                                help="Seed to initialize the network"
                                );
    arguments = arg_parser.parse_args();
    check_arguments( arguments );

    exit(   main(   combination = arguments.combination,
                    learning_rate = arguments.learning_rate,
                    epochs = arguments.epochs,
                    batches = arguments.batches,
                    seed = arguments.seed
                    )
            );

## EOF
