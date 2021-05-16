import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Splits the training set into a training set of size train_prop * 'original size'
# and a validation set with the remaining data points.
def split_train_set(train_X, train_Y, train_prop):
    assert train_X.shape[0] >= 2, 'Error while splitting the training set, make sure there are at least 2 items.'

    dataset = np.column_stack((train_X, train_Y))
    dataset = np.random.permutation(dataset)

    # TODO: Is it correct to force at least 1 item into the sets?
    new_train_X, new_train_Y = np.array([dataset[0][:-1]]), np.array([dataset[0][-1]])
    new_val_X, new_val_Y = np.array([dataset[1][:-1]]), np.array([dataset[1][-1]])
    
    for p in dataset[2:]:
        if np.random.rand() <= train_prop:
            new_train_X = np.vstack([new_train_X, p[:-1]])
            new_train_Y = np.vstack([new_train_Y, p[-1]])
        else:
            new_val_X = np.vstack([new_val_X, p[:-1]])
            new_val_Y = np.vstack([new_val_Y, p[-1]])

    return (new_train_X, new_train_Y, new_val_X, new_val_Y)

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
            out = 1 if line[0] == 1 else -1
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