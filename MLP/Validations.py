from MLP.Network import Sequential
from MLP.GradientDescent import gradient_descent
from MLP.Utils import change_seed, combine_subseed
from math import inf
import numpy as np

def holdout_hyperconfiguration(conf, training, validation, trial_subseed):
    model = Sequential(change_seed(conf, subseed=trial_subseed))
    result = gradient_descent(model, training, validation, conf)
    return {'best_epoch'          : result['best_epoch'],
            'best_train_error'    : result['train_errors']    [result['best_epoch']],
            'best_train_accuracy' : result['train_accuracies'][result['best_epoch']],
            'best_val_error'      : result['val_errors']      [result['best_epoch']],
            'best_val_accuracy'   : result['val_accuracies']  [result['best_epoch']],
            'best_metric_train_error' : result['metric_train_errors'][result['best_epoch']],
            'best_metric_val_error'   : result['metric_val_errors']  [result['best_epoch']],
            'plots'               : [result]
            }

def kfold_hyperconfiguration(conf, folded_dataset, trial_subseed):
    folds = []

    for f in range(len(folded_dataset)):
        model = Sequential(change_seed(conf, subseed=combine_subseed(f, trial_subseed)))

        validation = folded_dataset[f]
        training = np.concatenate(folded_dataset[:f] + folded_dataset[f + 1:])

        results = gradient_descent(model, training, validation, conf)
        folds.append(results)

    return {'best_epoch'          : np.average(np.array([f['best_epoch']                        for f in folds])),
            'best_train_error'    : np.average(np.array([f['train_errors']    [f['best_epoch']] for f in folds])),
            'best_train_accuracy' : np.average(np.array([f['train_accuracies'][f['best_epoch']] for f in folds])),
            'best_val_error'      : np.average(np.array([f['val_errors']      [f['best_epoch']] for f in folds])),
            'best_val_accuracy'   : np.average(np.array([f['val_accuracies']  [f['best_epoch']] for f in folds])),
            'best_metric_train_error' : np.average(np.array([f['metric_train_errors'][f['best_epoch']] for f in folds])),
            'best_metric_val_error'   : np.average(np.array([f['metric_val_errors']  [f['best_epoch']] for f in folds])),
            'plots'               : folds
           }
