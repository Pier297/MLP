from MLP.Validations import holdout_hyperconfiguration, kfold_hyperconfiguration
from itertools import repeat
from multiprocessing import Pool
from MLP.experiments.utils import argmin
from MLP.LossFunctions import loss_function_from_name
from MLP.Network import Sequential
from math import ceil
import numpy as np

def generate_hyperparameters(
        loss_function_name: str = "Cross Entropy",
        in_dimension    = 17,
        out_dimension   = 1,
        target_domain = (0, 1),
        validation_percentage=0.2,
        mini_batch_percentage=1.0,
        max_unlucky_epochs=10,
        max_epochs=250,
        validation_type = {'method': 'holdout'},
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
                    configurations.append({"in_dimension":          in_dimension,
                                           "out_dimension":         out_dimension,
                                           "target_domain":         target_domain,
                                           "validation_type":       validation_type,
                                           "loss_function_name":    loss_function_name,
                                           "lr":                    lr,
                                           "l2":                    l2,
                                           "momentum":              momentum,
                                           "hidden_layers":         hidden_layers,
                                           "validation_percentage": validation_percentage,
                                           "mini_batch_percentage": mini_batch_percentage,
                                           "max_unlucky_epochs":    max_unlucky_epochs,
                                           "max_epochs":            max_epochs,
                                           "seed":                  np.random.randint(2**31-1),
                                           "print_stats":           False})
    return configurations

def call_holdout(args):
    i, (conf, (train_set, val_set)) = args
    results = holdout_hyperconfiguration(conf, train_set, val_set)
    print(f"Holdout finished ({i}, val_error: {results['val_error']})")
    return results

def holdout_grid_search(hyperparameters, training, n_workers):
    # Splits the dataset into a validation set of size val_prop * 'original size'
    # and a training set with the remaining data points.
    def split_train_set(dataset_unshuffled, val_prop):
        # Shuffle the data
        dataset = np.random.permutation(dataset_unshuffled)
        val_size = int(val_prop * dataset)
        train_set = dataset[val_size:][:]
        val_set = dataset[:val_size][:]
        return (train_set, val_set)
    # Split the dataset into train and validation set.
    (train_set, val_set) = split_train_set(training, hyperparameters[0]['validation_percentage'])
    with Pool(processes=n_workers) as pool:
        return pool.map(call_holdout, enumerate(zip(hyperparameters, repeat((train_set, val_set)))))

def call_kfold(args):
    i, (conf, (folded_dataset)) = args
    results = kfold_hyperconfiguration(conf, folded_dataset)
    print(f"K-fold finished ({i}, val_error: {results['val_error']})")
    return results

def kfold_grid_search(hyperparameters, training, n_workers):
    def split_chunks(vals, k):
        size = ceil(len(vals)/k) 
        for i in range(0, len(vals), size):
            yield vals[i:i + size][:]

    folded_dataset = list(split_chunks(np.random.permutation(training), hyperparameters[0]['validation_type']['k']))
    with Pool(processes=n_workers) as pool:
        return pool.map(call_kfold, enumerate(zip(hyperparameters, repeat((folded_dataset)))))

def grid_search(hyperparameters, training, n_workers):
    if hyperparameters[0]["validation_type"]["method"] == 'holdout':
        validation_results = holdout_grid_search(hyperparameters, training, n_workers)
    elif hyperparameters[0]["validation_type"]["method"] == 'kfold':
        validation_results = kfold_grid_search(hyperparameters, training, n_workers)
    else:
        raise ValueError(f'Unknown validation_type: {hyperparameters[0]["validation_type"]["method"]}')

    # Find the best hyperparameters configuration
    best_i = argmin(lambda c: c['val_error'], validation_results)

    return hyperparameters[best_i], validation_results[best_i]
