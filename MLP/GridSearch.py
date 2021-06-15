from MLP.Validations import holdout, kfold
from itertools import repeat
from multiprocessing import Pool
from MLP.experiments.utils import split_train_set, argmin
from MLP.LossFunctions import loss_function_from_name
from MLP.Network import Sequential
from math import inf, ceil
import os
import time
import numpy as np

def generate_hyperparameters(
        loss_func: str = "Cross Entropy",
        in_dimension    = 17,
        out_dimension   = 1,
        target_domain = (0, 1),
        train_percentage=0.8,
        mini_batch_percentage=1.0,
        validation_type={'method': 'holdout'},
        lr_values = [0.4],
        l2_values = [0],
        momentum_values = [0],
        hidden_layers_values = [([('tanh',4)],'sigmoid')]
    ):
    configurations = []
    for lr in lr_values:
        for l2 in l2_values:
            for momentum in momentum_values:
                for hidden_layers in hidden_layers_values:
                    configurations.append({"in_dimension": in_dimension,
                                           "out_dimension": out_dimension,
                                           "target_domain": target_domain,
                                           "validation_type": validation_type,
                                           "loss_function": loss_func,
                                           "lr": lr, "l2": l2, "momentum": momentum,
                                           "hidden_layers": hidden_layers,
                                           "train_percentage": train_percentage,
                                           "mini_batch_percentage": mini_batch_percentage})
    return configurations


def call_holdout(args):
    conf, (train_set, val_set) = args

    #  How many times to repeat the training with the same configuration in order to reduce the validation error variance
    K = 3
    trials = []
    best_val_error = inf
    for t in range(K):
        results = holdout(conf, train_set, val_set, conf["target_domain"],
                          loss_function_from_name(conf["loss_function"]),
                          lr=conf["lr"], l2=conf["l2"], momentum=conf["momentum"], mini_batch_percentage=conf["mini_batch_percentage"], MAX_UNLUCKY_STEPS = 25, MAX_EPOCHS = 250)
        if results['val_error'] < best_val_error:
            best_val_error = results['val_error']
        trials.append(results)
 
    return {'val_error': best_val_error, 'trials': trials}


def call_kfold(args):

    conf, (folded_dataset) = args

    print('Validation started pid: ', str(os.getpid()))

    return kfold(conf, folded_dataset,
                 conf["target_domain"], loss_function_from_name(conf["loss_function"]),
                 conf["lr"], conf["l2"], conf["momentum"],
                 conf["mini_batch_percentage"], MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 250)


def run_holdout_grid_search(hyperparameters, training, n_workers):
    # Split the dataset into train and validation set.
    (train_set, val_set) = split_train_set(training, hyperparameters[0]['train_percentage'])
    with Pool(processes=n_workers) as pool:
        return pool.map(call_holdout, zip(hyperparameters, repeat((train_set, val_set))))
    #return map(validation_wrapper(call_holdout), zip(hyperparameters, repeat((train_set, val_set))))


def run_kfold_grid_search(hyperparameters, training, n_workers):
    def split_chunks(vals, k):
        size = ceil(len(vals)/k) 
        for i in range(0, len(vals), size):
            yield vals[i:i + size][:]
    folded_dataset = list(split_chunks(np.random.permutation(training), hyperparameters[0]['validation_type']['k']))
    with Pool(processes=n_workers) as pool:
        return pool.map(call_kfold, zip(hyperparameters, repeat((folded_dataset))))


def grid_search(hyperparameters, training, n_workers=1):
    if hyperparameters[0]["validation_type"]["method"] == 'holdout':
        validation_results = run_holdout_grid_search(hyperparameters, training, n_workers)
    elif hyperparameters[0]["validation_type"]["method"] == 'kfold':
        validation_results = run_kfold_grid_search(hyperparameters, training, n_workers)
    else:
        raise ValueError(f'Unknown validation_type: {hyperparameters[0]["validation_type"]["method"]}')

    # Find the best hyperparameters configuration
    best_i = argmin(lambda c: c['val_error'], validation_results)

    return hyperparameters[best_i], validation_results[best_i]
