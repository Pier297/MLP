import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 'number' can be 1, 2 or 3
# It returns the dataset splitted into training and test set.
# Already handling the one-hot representation
def load_monk(number: int):
    (train_X, train_Y) = _load_monk(f'MLP/experiments/data/monks-{str(number)}.train')
    (test_X, test_Y) = _load_monk(f'MLP/experiments/data/monks-{str(number)}.test')
    return (train_X, train_Y, test_X, test_Y)

def _load_monk(file_name: str):
    X, Y = [], []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.split(' ')
            # ['', '1', '3', '3', '2', '3', '4', '2', 'data_432\n']
            line = line[1:-1]
            # ['1', '3', '3', '2', '3', '4', '2']
            line = [int(x) for x in line]
            # [1, 3, 3, 2, 3, 4, 2]
            out = line[0]
            x1 = one_hot(line[1], 3)
            x2 = one_hot(line[2], 3)
            x3 = one_hot(line[3], 2)
            x4 = one_hot(line[4], 3)
            x5 = one_hot(line[5], 4)
            x6 = one_hot(line[6], 2)
            x = np.concatenate((x1, x2, x3, x4, x5, x6))
            y = [out]
            X.append(x)
            Y.append(y)
    return np.array(X), np.array(Y)

def one_hot(x, k):
    r = np.zeros(k, dtype=int)
    r[x - 1] = 1
    return r