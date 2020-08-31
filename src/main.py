"""
This file is the main function that parses user argument and runs experiments.

Date:
    August 15, 2020

Project:
    XAIGAN

Authors:
    name: Vineel Nagisetty, Laura Graves, Joseph Scott, Vijay Ganesh
    contact: vineel.nagisetty@uwaterloo.ca
"""

# imports

import argparse
import sys
import torch
[sys.path.append(i) for i in ['.', '..']]

from src.experiment_enums import experimentsCurrent


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run the experiment using regular and logic GANs")
    args = parser.parse_args()
    experiment_setup(args)


def experiment_setup(args: argparse.Namespace) -> None:
    """
    This function sets up the experiment and runs it for both regular and logic GANs
    :param args: dictionary arguments from user
    :return: None
    """
    torch.backends.cudnn.benchmark = True
    experiments = experimentsCurrent
    for experiment in experiments:
        experiment.run(logging_frequency=4)


if __name__ == "__main__":
    main()
