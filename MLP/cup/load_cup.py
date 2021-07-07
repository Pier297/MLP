import numpy as np

# Helper procedures to load the files globally

def load_cup():
    """
    Load the cup datasets from the globally available files.
    :return: training, test
    """
    training = load_cup_from_file('MLP/cup/data/ML-CUP-training.csv')
    test = load_cup_from_file('MLP/cup/data/ML-CUP-test.csv')

    return training, test

def load_blind_cup():
    """
    Load the bling cup inputs from the globally available files.
    :return: inputs
    """
    return load_blind_cup_from_file('MLP/cup/data/ML-CUP20-TS.csv')

# General procedures for file loading given the file names

def load_cup_from_file(file_name):
    """
    Load the cup datasets from the given filename.
    :return: training, test
    """
    data = []
    with open(file_name) as f:
        for line in f.readlines():
            # Parse the file as standard CSV
            line = line.strip().split(',')
            line = [float(x) for x in line]
            # Transform to nparray
            data.append(np.array(line))
    return np.array(data)

def load_blind_cup_from_file(file_name):
    """
    Load the blind cup inputs from the given filename.
    :return: inputs
    """
    data = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.strip().split(',')
            line = [float(x) for x in line]
            data.append(np.array(line))
    return np.array(data)