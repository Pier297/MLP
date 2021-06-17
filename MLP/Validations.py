from MLP.Network import Sequential
from MLP.LossFunctions import loss_function_from_name
from MLP.Optimizers import gradient_descent
from math import inf
from MLP.Network import reset
import numpy as np

def holdout_hyperconfiguration(conf, training, validation, number_trials=3):
    trials = []
    best_val_error = inf
    best_trial = None
    for t in range(number_trials):
        model = Sequential(conf, change_seed=True)
        #  How many times to repeat the training with the same configuration in order to reduce the validation error variance
        results = gradient_descent(model, training, validation, conf)
        if results['val_error'] < best_val_error:
            best_val_error = results['val_error']
            best_trial = results
            trials.append(results)
    return {'val_error': best_val_error, 'trials': trials, 'best_trial': best_trial}

def kfold_hyperconfiguration(conf, folded_dataset):
    average_val_error = 0.0
    trials = []

    for i in range(len(folded_dataset)):
        model = Sequential(conf, change_seed=True)

        validation = folded_dataset[i]
        training_list = folded_dataset[:i] + folded_dataset[i + 1:]
        training = np.concatenate(training_list)

        results = gradient_descent(model, training, validation, conf)
        trials.append(results)
        average_val_error += results["val_error"]

    return {'val_error': average_val_error / len(folded_dataset), 'trials': trials}