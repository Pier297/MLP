from MLP.Network import Sequential
from MLP.GradientDescent import gradient_descent
from MLP.Utils import change_seed, combine_subseed
from math import inf
import numpy as np

def holdout_hyperconfiguration(conf, training, validation, trial_subseed):
    model = Sequential(change_seed(conf, subseed=trial_subseed))
    return gradient_descent(model, training, validation, conf)


def pad_data_constantly(target_size, data_list):
    data = np.array(data_list)
    return np.concatenate((data, np.full((target_size - data.shape[0],), data[-1])))

def average_zip(results):
    return np.sum(np.stack(results),axis=0) / len(results)

def average_uneven_length_results(results):
    max_length = max(map(len, results))
    results_even = list(map(lambda data: pad_data_constantly(max_length, data), results))
    return average_zip(results_even)

def combine_fold_results(folds):
    get_all_np = lambda k, d: list(map(lambda x: np.array(x[k]), d))
    return {'best_val_error':   np.average(get_all_np('best_val_error', folds)),
            'best_epoch':       np.average(get_all_np('best_epoch', folds)),
            'train_errors':     average_uneven_length_results(get_all_np('train_errors',     folds)),
            'train_accuracies': average_uneven_length_results(get_all_np('train_accuracies', folds)),
            'val_errors':       average_uneven_length_results(get_all_np('val_errors',       folds)),
            'val_accuracies':   average_uneven_length_results(get_all_np('val_accuracies',   folds)),
            'watch_errors':     average_uneven_length_results(get_all_np('watch_errors',     folds)),
            'watch_accuracies': average_uneven_length_results(get_all_np('watch_accuracies', folds)),
            'weights_norms':    average_uneven_length_results(get_all_np('weights_norms',    folds)),
            'gradient_norms':   average_uneven_length_results(get_all_np('gradient_norms',   folds))}

def kfold_hyperconfiguration(conf, folded_dataset, trial_subseed):
    folds = []

    for f in range(len(folded_dataset)):
        model = Sequential(change_seed(conf, subseed=combine_subseed(f, trial_subseed)))

        validation = folded_dataset[f]
        training_list = folded_dataset[:f] + folded_dataset[f + 1:]
        training = np.concatenate(training_list)

        results = gradient_descent(model, training, validation, conf)
        folds.append(results)

    return combine_fold_results(folds)
