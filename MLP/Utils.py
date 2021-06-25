import random
import numpy as np
from math import inf

def argmin(f, v):
    minimum = inf
    minimum_i = -1
    for (i, x) in enumerate(v):
        xv = f(x)
        if minimum > xv:
            minimum = xv
            minimum_i = i
    if minimum_i == -1:
        raise 'Empty list passed to argmin'
    return minimum_i

def average(s):
    return sum(s) / len(s)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def generate_seed():
    return np.random.randint(2**16-1)

def combine_subseed(ss1, ss2):
    return (2**16 * ss1) + ss2 

def change_seed(d, subseed=None):
    return {**d, "seed": generate_seed() if subseed is None else d['seed'] + subseed}

def get_data_statistics(X):
    return {'avg': np.average(X), 'std': np.std(X)}

def normalize(X, stats):
    return (X - stats['avg']) / stats['std']

def denormalize(X, stats):
    return (X * stats['std']) + stats['avg']

def normalize_data(M_original, stats):
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = normalize(M[:, i], stat)
    return M

def denormalize_data(M_original, stats):
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = denormalize(M[:, i], stat)
    return M