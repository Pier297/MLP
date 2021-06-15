from MLP.Network import Sequential
from MLP.LossFunctions import loss_function_from_name
from MLP.Regularizers import early_stopping
from math import inf
from MLP.Network import reset
import numpy as np

def holdout(conf, training, validation, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage=1.0, MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 250):
    model = Sequential(conf)
    
    train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch = early_stopping(model, training, validation, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage, MAX_UNLUCKY_STEPS, MAX_EPOCHS)
    
    current_val_error = val_errors[early_best_epoch-1]
    results = {'val_error': current_val_error, 'epochs': early_best_epoch, 'train_errors': train_errors, 'val_errors': val_errors, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
 
    return results


def kfold(conf, folded_dataset, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage=1.0, MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 250):
    e = 0.0
    history = []
    for i in range(len(folded_dataset)):
        validation = folded_dataset[i]
        # TODO: Find way to directly do this with numpy methods
        training = np.ndarray((1, 18))
        for j, fold in enumerate(folded_dataset):
            if j != i:
                training = np.vstack([training, fold])

        training = training[1:][:]
        
        results = holdout(conf, training, validation, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage, MAX_UNLUCKY_STEPS, MAX_EPOCHS)
        history.append(results)
        e += results["val_error"]

    return {'val_error': e / len(folded_dataset), 'trials': history}