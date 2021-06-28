from MLP.Validations import holdout_hyperconfiguration, kfold_hyperconfiguration
from itertools import repeat
from multiprocessing import Pool
from MLP.Utils import argmin, change_seed, average
from MLP.LossFunctions import loss_function_from_name
from MLP.Network import Sequential
from multiprocessing import cpu_count
from math import ceil, inf
import time
import numpy as np

def generate_hyperparameters(params):
    results = [{}]
    def cartesian_product(k, vs, ds):
        for v_index, v in enumerate(vs):
            for d in ds:
                yield {**d, k: v}

    for k, v in params.items():
        if type(v) is list:
            results = list(cartesian_product(k, v, results))
        else:
            results = list(cartesian_product(k, [v], results))

    return list(map(change_seed, results))

def trialize(number_trials, validation_method):
    trials = []
    sum_val_errors = 0.0
    for t in range(number_trials):
        result = validation_method(t) # Give t as the subseed
        sum_val_errors += result['best_val_error']
        trials.append(result)
    return {'val_error': sum_val_errors / number_trials, 'trials': trials}

def call_holdout(args):
    try:
        i, (conf, (train_set, val_set)) = args
        before = time.perf_counter()
        results = trialize(conf['number_trials'],
                        lambda subseed: holdout_hyperconfiguration(conf, train_set, val_set, subseed))
        after = time.perf_counter()
        print(f"Holdout finished (VE: {results['val_error']}, avg trial best epoch: {int(average(list(map(lambda x: x['best_epoch'], results['trials']))))}), time (s): {after-before}")
        return results
    except Exception:
        print('Got exception with conf: ', conf)
        return {'val_error': inf}

def call_kfold(args):
    try:
        i, (conf, (folded_dataset)) = args
        before = time.perf_counter()
        results = trialize(conf['number_trials'],
                           lambda subseed: kfold_hyperconfiguration(conf, folded_dataset, subseed))
        after = time.perf_counter()
        print(f"K-fold finished (VE: {results['val_error']}, avg trial best epoch: {int(average(list(map(lambda x: x['best_epoch'], results['trials']))))}), time (s): {after-before}")
        return results
    except Exception:
        print('Got exception with conf: ', conf)
        return {'val_error': inf}


def holdout_grid_search(hyperparameters, training):
    # Splits the dataset into a validation set of size val_prop * 'original size'
    # and a training set with the remaining data points.
    def split_train_set(dataset_unshuffled, val_prop):
        # Shuffle the data
        dataset = np.random.permutation(dataset_unshuffled)
        val_size = int(val_prop * dataset.shape[0])
        train_set = dataset[val_size:][:]
        val_set = dataset[:val_size][:]
        return (train_set, val_set)
    # Split the dataset into train and validation set.
    (train_set, val_set) = split_train_set(training, hyperparameters[0]['validation_percentage'])
    with Pool(processes=cpu_count()) as pool:
        return pool.map(call_holdout, enumerate(zip(hyperparameters, repeat((train_set, val_set)))))

def kfold_grid_search(hyperparameters, training):
    def split_chunks(vals, k):
        size = ceil(len(vals)/k)
        for i in range(0, len(vals), size):
            yield vals[i:i + size][:]

    folded_dataset = list(split_chunks(np.random.permutation(training), hyperparameters[0]['validation_type']['k']))
    with Pool(processes=cpu_count()) as pool:
        return pool.map(call_kfold, enumerate(zip(hyperparameters, repeat((folded_dataset)))))

def grid_search(hyperparameters, training):
    if hyperparameters[0]["validation_type"]["method"] == 'holdout':
        validation_results = holdout_grid_search(hyperparameters, training)
    elif hyperparameters[0]["validation_type"]["method"] == 'kfold':
        validation_results = kfold_grid_search(hyperparameters, training)
    else:
        raise ValueError(f'Unknown validation_type: {hyperparameters[0]["validation_type"]["method"]}')

    # Find the best hyperparameters configuration
    best_i = argmin(lambda c: c['val_error'], validation_results)

    return hyperparameters[best_i], validation_results[best_i]
