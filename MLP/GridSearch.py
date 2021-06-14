from MLP.Validations import holdout, kfold
from itertools import repeat
from multiprocessing import Pool
from MLP.experiments.utils import split_train_set
from MLP.LossFunctions import loss_function_from_name
from MLP.Network import Sequential
from math import inf
import os
import time

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
    start = time.time()
    conf, (train_set, val_set) = args
    loss_func = loss_function_from_name(conf["loss_function"])

    #  How many times to repeat the training with the same configuration in order to reduce the validation error variance
    K = 3
    best_results = {'val_error': inf, 'epochs': 0, 'train_errors': [], 'val_errors': [], 'train_accuracies': [], 'val_accuracies': []}
    for t in range(K):
        model = Sequential(conf)

        results = holdout(model, train_set, val_set, conf["target_domain"], loss_func, lr=conf["lr"], l2=conf["l2"], momentum=conf["momentum"], mini_batch_percentage=conf["mini_batch_percentage"], MAX_UNLUCKY_STEPS = 25, MAX_EPOCHS = 250)
        # TODO: imo we should take an avg of the val error over the K runs otherwise it's all based on the lucky random init
        #       and then we can plot the learning curves with the deviations..
        if results['val_error'] < best_results['val_error']:
            best_results = results

    end = time.time()

    print('Finished', str(os.getpid()), end-start, len(best_results["val_errors"]))
 
    return best_results


def call_kfold(args):

    print('Finished')


def run_holdout_grid_search(hyperparameters, training, n_workers=1):
    # Split the dataset into train and validation set.
    (train_set, val_set) = split_train_set(training, hyperparameters[0]['train_percentage'])
    with Pool(processes=n_workers) as pool:
        return pool.map(call_holdout, zip(hyperparameters, repeat((train_set, val_set))))
    #return map(call_holdout, zip(hyperparameters, repeat((train_set, val_set))))


def run_kfold_grid_search(hyperparameters, training, n_workers=1):
    folded_dataset = 1 # TODO
    with Pool(processes=n_workers) as pool:
        return pool.map(call_kfold, zip(hyperparameters, repeat((folded_dataset))))


def grid_search(hyperparameters, training, n_workers=1):
    if hyperparameters[0]["validation_type"]["method"] == 'holdout':
        return run_holdout_grid_search(hyperparameters, training, n_workers=1)
    elif hyperparameters[0]["validation_type"]["method"] == 'kfold':
        return run_kfold_grid_search(hyperparameters, training, n_workers=1)
    else:
        raise ValueError(f'Unknown validation_type: {hyperparameters[0]["validation_type"]["method"]}')