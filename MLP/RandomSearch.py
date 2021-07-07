from MLP.Utils import change_seed
import numpy as np

def sample(v):
    if v['method'] == 'uniform':
        return np.random.uniform(v['a'], v['b'])
    elif v['method'] == 'normal':
        return np.random.normal(v['center'], v['b'] - v['a'])
    else:
        raise ValueError(f'Unknown sample method: {v["method"]}')

def gen_range(chosen, space, method='uniform', boundaries=(0.00001, 0.99)):
    (a, b) = (chosen - (chosen*0.1), chosen + (chosen*0.1))

    if a < boundaries[0]:
        a = boundaries[0]
    
    if b > boundaries[1]:
        b = boundaries[1]

    return {'random': True, 'a': a, 'b': b, 'center': chosen, 'method': method}

def generate_hyperparameters_random(master, params, generations=100):
    def stream():
        for _ in range(generations):
            instance = {}
            for k, v in params.items():
                if type(v) is dict and 'random' in v:
                    instance[k] = sample(v)
                else:
                    instance[k] = v
            yield instance
    return [master] + list(map(change_seed, stream()))
