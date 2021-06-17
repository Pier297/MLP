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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 'number' can be 1, 2 or 3
# It returns the dataset splitted into training and test set.
# Already handling the one-hot representation
def load_monk(number: int, target_domain=(-1, 1)):
    training = _load_monk(f'MLP/experiments/data/monks-{str(number)}.train', target_domain)
    test = _load_monk(f'MLP/experiments/data/monks-{str(number)}.test', target_domain)
    return (training, test)

def _load_monk(file_name: str, target_domain=(-1, 1)):
    data = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.split(' ')
            # ['', '1', '3', '3', '2', '3', '4', '2', 'data_432\n']
            line = line[1:-1]
            # ['1', '3', '3', '2', '3', '4', '2']
            line = [int(x) for x in line]
            # [1, 3, 3, 2, 3, 4, 2]
            out = target_domain[1] if line[0] == 1 else target_domain[0]
            x1 = one_hot(line[1], 3)
            x2 = one_hot(line[2], 3)
            x3 = one_hot(line[3], 2)
            x4 = one_hot(line[4], 3)
            x5 = one_hot(line[5], 4)
            x6 = one_hot(line[6], 2)
            x = np.concatenate((x1, x2, x3, x4, x5, x6))
            y = [out]
            data.append(np.concatenate([x, y]))
    return np.array(data)

def one_hot(x, k):
    r = np.zeros(k, dtype=int)
    r[x - 1] = 1
    return r