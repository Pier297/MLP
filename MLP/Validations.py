from MLP.Network import Sequential
from MLP.LossFunctions import loss_function_from_name
from MLP.Regularizers import early_stopping
from math import inf


def holdout(model, training, validation, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage=1.0, MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 250):
    train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch = early_stopping(model, training, validation, target_domain, loss_func, lr, l2, momentum, mini_batch_percentage, MAX_UNLUCKY_STEPS, MAX_EPOCHS)
    
    current_val_error = val_errors[early_best_epoch-1]
    results = {'val_error': current_val_error, 'epochs': early_best_epoch, 'train_errors': train_errors, 'val_errors': val_errors, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
 
    return results


def kfold(args):
    conf, (folded_dataset, in_dimension, out_dimension, target_domain) = args

    model = Sequential(conf, in_dimension, out_dimension)

    loss_func = loss_function_from_name(conf["loss_function"])

    for i in range(len(folded_dataset)):
        validation = folded_dataset[i]
        training = [fold for fold, j in enumerate(folded_dataset) if i != j]

        (train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch) = early_stopping(model, loss_func, conf["lr"], conf["l2"], conf["momentum"], conf["train_percentage"], training, validation, MAX_UNLUCKY_STEPS=50, MAX_EPOCHS=500, target_domain=target_domain)

    return train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch