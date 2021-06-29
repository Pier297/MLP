import numpy as np

def sample(v):
    if v['method'] == 'uniform':
        return np.random.uniform(v['a'], v['b'])
    elif v['method'] == 'normal':
        return np.random.normal(v['center'], v['b'] - v['a'])
    else:
        raise ValueError(f'Unknown sample method: {v["method"]}')

def gen_range(chosen, space, method='uniform'):
    i = space.index(chosen)

    if len(space) == 1:
        (a, b) = (space[0] - space[0]/2, space[0] + space[0]/2)
    elif i == 0:
        (a, b) = (space[0] - (space[1] - space[0]), space[1])
    elif i == len(space)-1:
        (a, b) = (space[-2], space[-1] + (space[-1] - space[-2]))
    else:
        (a, b) = (space[i-1], space[i+1])

    return {'random': True, 'a': a, 'b': b, 'center': chosen, 'method': method}

def generate_hyperparameters_random(params, generations=100):
    def stream():
        for _ in range(generations):
            instance = {}
            for k, v in params.items():
                if type(v) is dict and 'random' in v:
                    instance[k] = sample(v)
                else:
                    instance[k] = v
            print("Lr generated:", instance['lr'])
            yield instance
    return list(stream())
