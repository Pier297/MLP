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

# GridSearch.py

# Main definitions for the GridSearch and RandomSearch hyperparameters list exploration procedures.

def generate_hyperparameters(params):
    """
    Generate a list of hyperparameters given all the possibilities in params.
    If the argument is a list, it is generated with all the possibilities.
    Otherwise, it is simply repeated as-is.
    :param params: dictionary with all the hyperparameters to combine
    :return: a list of all the possible configurations (cartesian product)
    """
    results = [{}]

    # Helper function to apply the cartesian product on a given key with values vs
    def cartesian_product(k, vs, ds):
        for v_index, v in enumerate(vs):
            for d in ds:
                yield {**d, k: v}

    # For all hyperparameters,
    for k, v in params.items():
        # Generate the combination with the values if it is a list
        if type(v) is list:
            results = list(cartesian_product(k, v, results))
        else:
            results = list(cartesian_product(k, [v], results))

    return list(map(change_seed, results))

def trialize(number_trials, validation_method):
    """
    Repeat the given validation_method the given number of times, and produce
    and averaged list of all the values returned by the validation method.
    :param validation_method: anonymous function to run the CV
    :return: dictionary of results with the same signature of the CV function
    """

    # Run all the trials

    trials = [validation_method(t) for t in range(number_trials)]

    # Extract all the relevant values to be averaged in the trials

    best_val_errors          = np.array([t['best_val_error'] for t in trials])
    best_train_errors        = np.array([t['best_train_error'] for t in trials])
    best_metric_val_errors   = np.array([t['best_metric_val_error'] for t in trials])
    best_metric_train_errors = np.array([t['best_metric_train_error'] for t in trials])
    best_trial_plots         = argmin(lambda x: x['best_val_error'], trials)['plots']
    trials_epochs            = np.array([t['best_epoch'] for t in trials])

    return {'val_error'                    : np.average(best_val_errors),
            'train_error'                  : np.average(best_train_errors),
            'metric_val_error'             : np.average(best_metric_val_errors),
            'metric_train_error'           : np.average(best_metric_train_errors),
            # Compute the variance
            'val_error_var'                : np.var(best_val_errors),
            'train_error_var'              : np.var(best_train_errors),
            'metric_val_error_var'         : np.var(best_metric_val_errors),
            'metric_train_error_var'       : np.var(best_metric_train_errors),
            # Compute the general results and return all the trial results as they are
            'epochs'                       : ceil(np.average(trials_epochs)),
            'trials'                       : trials,
            'best_trial_plots'             : best_trial_plots,
            # Compute the results for each trial
            'trials_best_train_errors'     : np.array([t['best_train_error']    for t in trials]),
            'trials_best_train_accuracies' : np.array([t['best_train_accuracy'] for t in trials]),
            'trials_best_val_errors'       : np.array([t['best_val_error']      for t in trials]),
            'trials_best_val_accuracies'   : np.array([t['best_val_accuracy']   for t in trials]),
            }

def call_holdout(args):
    """
    Helper function required by holdout_grid_search to be parallelized.
    :param args: a tuple with configuration and the current datasets.
    :return: the holdout results given as standard CV dictionary
    """
    try:
        i, (conf, (train_set, val_set)) = args
        return trialize(conf['number_trials'],
                           lambda subseed: holdout_hyperconfiguration(conf, train_set, val_set, subseed))
    except Exception as e:
        return {'val_error': inf}

def call_kfold(args):
    """
    Helper function required by kfold_grid_search to be parallelized.
    :param args: a tuple with configuration and the current datasets.
    :return: the k-fold results given as standard CV dictionary
    """
    try:
        i, (conf, (folded_dataset)) = args
        return trialize(conf['number_trials'],
                           lambda subseed: kfold_hyperconfiguration(conf, folded_dataset, subseed))
    except Exception as e:
        return {'val_error': inf}

def holdout_grid_search(hyperparameters, training):
    """
    Main holdout procedure definition.
    :param hyperparameters: a list of all the configurations to be tested in the grid search.
    :return: a list with the validation results for each hyperconfiguration
    """

    # Helper function to perform the main holdout split, given the
    # validation proportions of the dataset to be split.
    # Note: this is only performed once and for all hyperconfigurations.

    def split_train_set(dataset, val_prop):
        val_size = int(val_prop * dataset.shape[0])
        train_set = dataset[val_size:][:]
        val_set = dataset[:val_size][:]
        return (train_set, val_set)

    # Split the dataset

    (train_set, val_set) = split_train_set(np.random.permutation(training), hyperparameters[0]['validation_percentage'])

    # Run the grid search in parallel:

    with Pool(processes=cpu_count()) as pool:
        return pool.map(call_holdout, enumerate(zip(hyperparameters, repeat((train_set, val_set)))))

def kfold_grid_search(hyperparameters, training):
    """
    Main holdout procedure definition.
    :param hyperparameters: a list of all the configurations to be tested in the grid search.
    :return: a list with the validation results for each hyperconfiguration
    """

    # Helper function to perform the main kfold splits, given the
    # k chunks to split the dataset in
    # Note: this is only performed once and for all hyperconfigurations.

    def split_chunks(vals, k):
        size = ceil(len(vals)/k)
        for i in range(0, len(vals), size):
            yield vals[i:i + size][:]

    # Split the dataset into the k folds

    folded_dataset = list(split_chunks(np.random.permutation(training), hyperparameters[0]['validation_type']['k']))

    # Run the grid search in parallel:

    with Pool(processes=cpu_count()) as pool:
        return pool.map(call_kfold, enumerate(zip(hyperparameters, repeat((folded_dataset)))))

def grid_search(hyperparameters, training):
    """
    Main grid search definition.
    The validation type is selected here, so that it can be run in parallel.
    (Note: the need for selecting the CV at this point and having a separate function for the two algorithms
    is a restriction given by the lack of support of first-class functions and their serializations from the
    multiprocessing library.)

    :param hyperparameters: a list of all the configurations to be tested in the grid search.
    :param training: the entire dataset (training + validation)
    :return: the best hyperconfiguration found, its results, the results of the entire search
             (this last element is needed to select the best k models)
    """

    # Select the CV type and run its parallel search

    if hyperparameters[0]["validation_type"]["method"] == 'holdout':
        validation_results = holdout_grid_search(hyperparameters, training)
    elif hyperparameters[0]["validation_type"]["method"] == 'kfold':
        validation_results = kfold_grid_search(hyperparameters, training)
    else:
        raise ValueError(f'Unknown validation_type: {hyperparameters[0]["validation_type"]["method"]}')

    # Find the best hyperparameters configuration by the minimum validation error

    best_i = argmin_index(lambda c: c['val_error'], validation_results)

    return hyperparameters[best_i], validation_results[best_i], validation_results
