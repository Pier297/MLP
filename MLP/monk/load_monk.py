import numpy as np

# load_monk.py

# Helper procedures to load the MONK's datasets.

def one_hot(x, k):
    """Returns the one-of-k encoding of the given variable, with k number of classes, in the range (-1,+1)."""
    r = np.zeros(k, dtype=int) - 1
    r[x - 1] = 1
    return r

def load_monk(number: int, target_domain=(-1, 1)):
    """Load the MONK dataset from the global filenames."""
    training = _load_monk(f'MLP/monk/data/monks-{str(number)}.train', target_domain)
    test = _load_monk(f'MLP/monk/data/monks-{str(number)}.test', target_domain)
    return (training, test)

def _load_monk(file_name: str, target_domain=(-1, 1)):
    """Load the MONK dataset from the given filename."""
    data = []
    with open(file_name) as f:
        for line in f.readlines():

            # Line hard-coded examples for the MONK dataset

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
