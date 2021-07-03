import random
import numpy as np
from math import inf, isnan

def argmin_index(f, v):
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
    return v[argmin_index(f, v)]

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

def data_statistics(dataset):
    return [column_statistics(col) for col in dataset.T]

def column_statistics(X):
    return {'avg': np.average(X), 'std': np.std(X), 'max': np.max(X), 'min': np.min(X)}

def normalize_column(X, stats):
    return X - stats['avg']

def denormalize_column(X, stats):
    return X + stats['avg']

def normalize(M_original, stats):
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = normalize_column(M[:, i], stat)
    return M

def denormalize(M_original, stats):
    M = np.array(M_original)
    for i, stat in enumerate(stats):
        M[:, i] = denormalize_column(M[:, i], stat)
    return M
