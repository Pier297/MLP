import numpy as np

def load_cup():
    training = []
    test = []
    training = _load_cup('MLP/cup/data/ML-CUP-training.csv')
    test = _load_cup('MLP/cup/data/ML-CUP-test.csv')

    return training, test
    

def _load_cup(file_name):
    data = []
    with open(file_name) as f:
        for line in f.readlines():
            line = line.strip().split(',')
            # ['-1.227729', '0.740105', '0.453528', '-0.761051', '-0.537705', '1.471803', '-1.143195', '2.034647', '1.603978', '-1.399807', '58.616635', '-36.878797']
            line = [float(x) for x in line]
            # [-1.227729, 0.740105, 0.453528, -0.761051, -0.537705, 1.471803, -1.143195, 2.034647, 1.603978, -1.399807, 58.616635, -36.878797]
            data.append(np.array(line))
    return np.array(data)