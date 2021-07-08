import random
import numpy as np
from math import inf, isnan

# Utils.py

# Various simple and general utility functions used throughout the code.

def argmin_index(f, v):
    """Find the index i of the vector minimizing the value of f(v[i])."""
    minimum = inf
    minimum_i = -1
    for (i, x) in enumerate(v):
        xv = f(x)
        if minimum >= xv:
            minimum = xv
            minimum_i = i
    if minimum_i == -1:
        print(v)
        raise 'Empty list passed to argmin'
    return minimum_i

def argmin(f, v):
    """Find the value x of the vector minimizing the value of f(x)."""
    return v[argmin_index(f, v)]

def average(s):
    """Python lists average (does not work for nparray)."""
    return sum(s) / len(s)

def set_seed(seed):
    """Wrapper on the number generation for both standard Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)

def generate_seed():
    """Generate a new predictable seed in the default range."""
    return np.random.randint(2**16-1)

def combine_subseed(ss1, ss2):
    """Deterministically combine two seeds to produce a third one."""
    return (2**16 * ss1) + ss2

def change_seed(d, subseed=None):
    """Inject a new seed inside an hyperconfiguration, returning a new one."""
    return {**d, "seed": generate_seed() if subseed is None else d['seed'] + subseed}

def data_statistics(dataset):
    """Compute the data statistics for each feature of the dataset."""
    return [column_statistics(col) for col in dataset.T]

def column_statistics(X):
    """Compute the data statistics for the given feature."""
    return {'avg': np.average(X), 'std': np.std(X), 'max': np.max(X), 'min': np.min(X)}

def normalize_column(X, stats):
    """Normalization procedure: simply center the feature."""
    return X - stats['avg']

def denormalize_column(X, stats):
    """Denormalization procedure: simply invert the feature centering using the same value."""
    return X + stats['avg']

def normalize(M_original, stats):
    """Normalize the given data by producing a new normalized copy such that all features are normalized."""
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = normalize_column(M[:, i], stat)
    return M

def denormalize(M_original, stats):
    """Denormalize the given data by producing a denormalized copy such that all features are returned as original."""
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = denormalize_column(M[:, i], stat)
    return M
