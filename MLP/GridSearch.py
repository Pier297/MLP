from MLP.Validations import holdout_hyperconfiguration, kfold_hyperconfiguration
from itertools import repeat
from multiprocessing import Pool
from MLP.Utils import argmin_index, argmin, change_seed, average
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
    trials = [validation_method(t) for t in range(number_trials)]
    best_val_errors = np.array([t['best_val_error'] for t in trials])
    best_trial_plots = argmin(lambda x: x['best_val_error'], trials)['plots']
    trials_epochs = np.array([t['best_epoch'] for t in trials])

    return {'val_error'                    : np.average(best_val_errors),
            'epochs'                       : ceil(np.average(trials_epochs)),
            'trials'                       : trials,
            'best_trial_plots'             : best_trial_plots,
            'trials_best_train_errors'     : np.array([t['best_train_error']    for t in trials]),
            'trials_best_train_accuracies' : np.array([t['best_train_accuracy'] for t in trials]),
            'trials_best_val_errors'       : np.array([t['best_val_error']      for t in trials]),
            'trials_best_val_accuracies'   : np.array([t['best_val_accuracy']   for t in trials]),
            }

def call_holdout(args):
    try:
        i, (conf, (train_set, val_set)) = args
        before = time.perf_counter()
        results = trialize(conf['number_trials'],
                           lambda subseed: holdout_hyperconfiguration(conf, train_set, val_set, subseed))
        after = time.perf_counter()
        print(f"Holdout finished (VE: {results['val_error']}), time (s): {after-before}")
        return results
    except Exception as e:
        print('Got exception with conf: ', conf, e.with_backtrace())
        return {'val_error': inf}

def call_kfold(args):
    try:
        i, (conf, (folded_dataset)) = args
        before = time.perf_counter()
        results = trialize(conf['number_trials'],
                           lambda subseed: kfold_hyperconfiguration(conf, folded_dataset, subseed))
        after = time.perf_counter()
        print(f"K-fold finished (VE: {results['val_error']}), time (s): {after-before}")
        return results
    except Exception as e:
        print('Got exception with conf: ', conf, e.with_backtrace())
        return {'val_error': inf}

def holdout_grid_search(hyperparameters, training):
    def split_train_set(dataset, val_prop):
        val_size = int(val_prop * dataset.shape[0])
        train_set = dataset[val_size:][:]
        val_set = dataset[:val_size][:]
        return (train_set, val_set)

    (train_set, val_set) = split_train_set(np.random.permutation(training), hyperparameters[0]['validation_percentage'])
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
    best_i = argmin_index(lambda c: c['val_error'], validation_results)

    return hyperparameters[best_i], validation_results[best_i], validation_results
