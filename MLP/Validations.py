from MLP.Network import Sequential
from MLP.GradientDescent import gradient_descent
from MLP.Utils import change_seed, combine_subseed
from math import inf
import numpy as np

# Validations.py

# Contains the core definitions of the holdout and k-fold procedures applied to each hyperparameter.

def holdout_hyperconfiguration(conf, training, validation, trial_subseed):
    """
    Main Cross-Validation Holdout per-hyperconfiguration procedure.
    This function will be called by the Holdout wrapper, which already
    applies the dataset the split once for all the grid search.
    This function treats a single hyperconfiguration.
    In the case of Holdout, simply train the network with the given training and validation datasets,
    where gradient descent already supports early stopping.
    :param conf: hyperparameters configuration
    :param training: training dataset
    :param validation: validation dataset
    :param trial_subseed: subseed induced by the different retinitialization trials
    """

    # Simply initialize and then train the network on the given training and validation.

    model = Sequential(change_seed(conf, subseed=trial_subseed))
    result = gradient_descent(model, training, validation, conf)
    return {'best_epoch'              : result['best_epoch'],
            'best_train_error'        : result['train_errors']       [result['best_epoch']],
            'best_train_accuracy'     : result['train_accuracies']   [result['best_epoch']],
            'best_val_error'          : result['val_errors']         [result['best_epoch']],
            'best_val_accuracy'       : result['val_accuracies']     [result['best_epoch']],
            'best_metric_train_error' : result['metric_train_errors'][result['best_epoch']] if conf['additional_metric'] is not None else -1,
            'best_metric_val_error'   : result['metric_val_errors']  [result['best_epoch']] if conf['additional_metric'] is not None else -1,
            'plots'                   : [result]
            }

def kfold_hyperconfiguration(conf, folded_dataset, trial_subseed):
    """
    Main Cross-Validation k-fold per-hyperconfiguration procedure.
    This function will be called by the k-fold wrapper, which already
    applies the dataset the split once for all the grid search.
    This function treats a single hyperconfiguration.
    In the case of k-fold, train the network for each fold provided
    and then compute the cross validation results by averaging the
    results obtained with each fold, and simply providing the folds as plots.
    :param conf: hyperparameters configuration
    :param training: training dataset
    :param validation: validation dataset
    :param trial_subseed: subseed induced by the different retinitialization trials
    """
    folds = []

    # For each fold,

    for f in range(len(folded_dataset)):

        # Construct the training dataset with the remaining fold data,

        validation = folded_dataset[f]
        training = np.concatenate(folded_dataset[:f] + folded_dataset[f + 1:])

        # and then train the network on the constructed dataset.

        model = Sequential(change_seed(conf, subseed=combine_subseed(f, trial_subseed)))
        results = gradient_descent(model, training, validation, conf)
        folds.append(results)

    # Combine the results by averaging the results.

    return {'best_epoch'              : np.average(np.array([f['best_epoch']                           for f in folds])),
            'best_train_error'        : np.average(np.array([f['train_errors']       [f['best_epoch']] for f in folds])),
            'best_train_accuracy'     : np.average(np.array([f['train_accuracies']   [f['best_epoch']] for f in folds])),
            'best_val_error'          : np.average(np.array([f['val_errors']         [f['best_epoch']] for f in folds])),
            'best_val_accuracy'       : np.average(np.array([f['val_accuracies']     [f['best_epoch']] for f in folds])),
            'best_metric_train_error' : np.average(np.array([f['metric_train_errors'][f['best_epoch']] if conf['additional_metric'] is not None else -1 for f in folds])),
            'best_metric_val_error'   : np.average(np.array([f['metric_val_errors']  [f['best_epoch']] if conf['additional_metric'] is not None else -1 for f in folds])),
            'plots'                   : folds
           }
