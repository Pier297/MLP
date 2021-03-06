from MLP.Utils import change_seed
import numpy as np

# RandomSearch.py

# This file contains the main definitions required to
# implement the random search in an extensible and hyperparameter-agnostic way.

def sample(v):
    """
    Simple helper function that generates the value with the given
    ranges, passed as a simple dictionary with ranges 'a' and 'b'.
    Optionally, a center parameter can be given for the case of gaussian
    generation.
    :param v: range dictionary
    :return: randomly perturbated value
    """
    if v['method'] == 'uniform':
        return np.random.uniform(v['a'], v['b'])
    elif v['method'] == 'normal':
        return np.random.normal(v['center'], v['b'] - v['a'])
    else:
        raise ValueError(f'Unknown sample method: {v["method"]}')

def gen_range(chosen, space, method='uniform', boundaries=(0.00001, 0.99), perturbation=0.15):
    """
    Wrapper function to encapsulate the desired hyperparameter variation.
    Returns a random range information compatibile with the sample function.
    :param chosen: value chosen for the given hyperparameter
    :param method: value generation method (either 'uniform' or 'normal')
    :param boundaries: correctness boundaries for the given hyperparameters
    :param perturbation: maximum percentage for parameter perturbation (default is 15%)
    :return: dictionary information for the range generation
    """

    # Identify the two ranges with perturbation% variation

    (a, b) = (chosen - (chosen*perturbation), chosen + (chosen*perturbation))

    # Clamp the given value within the boundaries

    if a < boundaries[0]:
        a = boundaries[0]

    if b > boundaries[1]:
        b = boundaries[1]

    # Finally, return the dictionary information with the constructed values

    return {'random': True, 'a': a, 'b': b, 'center': chosen, 'method': method}

def generate_hyperparameters_random(master, params, generations=100):
    """
    Random search equivalent of the generate_hyperparameters function defined in GridSearch.py.
    This function simply returns both a list with all the original
    hyperparameters along with #generations of additional perturbations of the original
    configuration for each desired parameter. The parameters that require perturbation must be
    encapsulated with the gen_range function; all other values will simply be returned as-is.
    :param master: original hyperconfiguration
    :param params: configuration containing the desired perturbation dictionaries
    :param generations: number of possibilities generated by the grid search
    :return: list of generations+1 hyperparameters configurations
    """
    def stream():
        # For each generation,
        for _ in range(generations):
            instance = {}
            # Perturb all the values of the configuration presenting a random dictionary object,
            # in the form constructed by the gen_range function.
            for k, v in params.items():
                if type(v) is dict and 'random' in v:
                    instance[k] = sample(v)
                else:
                    instance[k] = v
            yield instance

    # Regenerate the seed for each hyperconfiguration, as it is done in generate_hyperparameters.

    return [master] + list(map(change_seed, stream()))
