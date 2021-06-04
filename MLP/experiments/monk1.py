from MLP.Network import Sequential
from MLP.Optimizers import Gradient_descent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE, accuracy, loss_function_from_name
from MLP.GridSearch import generate_hyperparameters
from MLP.experiments.utils import load_monk, set_seed, split_train_set
from multiprocessing import Pool, cpu_count
from MLP.Regularizers import early_stopping
from MLP.ActivationFunctions import Tanh, Sigmoid
import numpy as np
from itertools import repeat
from math import inf

set_seed(2)

def kfold_grid_search(hyperparameters, training, activations_functions_names, target_domain, k):

    folded_dataset = 1 # TODO

    with Pool(processes=cpu_count()) as pool: # For each hyperparameter
       results = pool.map(kfold, zip(hyperparameters, repeat((folded_dataset, target_domain))))

def kfold(args):
    conf, (folded_dataset, in_dimension, out_dimension, target_domain) = args

    model = Sequential(conf, in_dimension, out_dimension)

    loss_func = loss_function_from_name(conf["loss_func"])

    for i in range(len(folded_dataset)):
        validation = folded_dataset[i]
        training = [fold for fold, j in enumerate(folded_dataset) if i != j]

        (train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch) = early_stopping(model, loss_func, conf["lr"], conf["l2"], conf["momentum"], conf["batch_percentage"], training, validation, MAX_UNLUCKY_STEPS=50, MAX_EPOCHS=500, target_domain=target_domain)

    return train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch



################################################

# def holdout_grid_search(hyperparameters, training, activations_functions_names, target_domain):
#     return 1
# 
# def holdout(args):
#     conf, (training, target_domain) = args
#     in_dimension  = training_data_X[0].shape[0]
#     out_dimension = training_data_Y[0].shape[0]
# 
#     model = Sequential(conf, in_dimension, out_dimension)
# 
#     loss_func = loss_function_from_name(conf["loss_func"])
# 
#     (train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch) = early_stopping(model, loss_func, conf["lr"], conf["l2"], conf["momentum"], conf["batch_percentage"], train_X, train_Y, validation, MAX_UNLUCKY_STEPS=50, MAX_EPOCHS=500, target_domain=target_domain)
# 
#     return train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch
# 
# def run_grid_search(args):
#     # Train the net on 'conf' and return (best_val_error, best_epoch)
# 
#     loss_func = loss_function_from_name(conf["loss_func"])
# 
#     # How many times to repeat the training with the same configuration in order to reduce the validation error variance
#     K = 3
#     best_config = {'val_error': inf, 'epochs': 0, 'train_errors': [], 'val_errors': [], 'train_accuracies': [], 'val_accuracies': []}
#     for t in range(K):
#         train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch = validation_function(conf, training, activations_functions_names, target_domain)
#         current_val_error = val_errors[early_best_epoch-1]
#         if current_val_error < best_config['val_error']:
#             best_config = {'val_error': current_val_error, 'epochs': early_best_epoch, 'train_errors': train_errors, 'val_errors': val_errors, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
# 
#     return best_config

################################################



if __name__ == '__main__':
    target_domain=(0, 1)

    (training, test) = load_monk(1, target_domain)

    hyperparameters = generate_hyperparameters(
        loss_func_values = ["Cross Entropy"],
        lr_values = [0.4],
        l2_values = [0],
        momentum_values = [0],
        hidden_layers_values = [[4]],
        hidden_layers_activations = [['tanh', 'sigmoid']],
        batch_percentage = [1.0]
    )

    validation_type = 'kfold'
    kfold_k         = 5
    in_dimension    = 17
    out_dimension   = 1

    if validation_type == 'holdout':
        results = holdout(hyperparameters, training, in_dimension, out_dimension, target_domain)
    elif validation_type == 'kfold':
        results = kfold(hyperparameters, training, in_dimension, out_dimension, target_domain, k=kfold_k)
    else:
        raise ValueError(f'Unknown validation type: {validation_type}')

    best_val_error = inf
    best_epochs = 0
    idx = -1
    # Search for the hyperparameter conf. with the lowest validation error
    for i, config in enumerate(results):
        val_error = config['val_error']
        if val_error < best_val_error:
            best_val_error = val_error
            best_epochs = config['epochs']
            idx = i

    best_hyperparameters = hyperparameters[idx]

    # --- Retraining: refine a new model with the best conf. and train on all the data ---

    model = Sequential(best_hyperparameters, in_dimension=in_dimension, out_dimension=out_dimension)

    (train_errors, train_accuracies, val_errors, val_accuracies) = Gradient_descent(model, training, np.array([]), loss_func=loss_function_from_name(best_hyperparameters['loss_function']), lr=best_hyperparameters["lr"], l2=best_hyperparameters["l2"], momentum=best_hyperparameters["momentum"], batch_percentage=best_hyperparameters["batch_percentage"], MAX_EPOCHS=best_epochs, target_domain=target_domain)

    print("Train accuracy =", accuracy(model, training, target_domain))
    #print("Test accuracy =", accuracy(model, test, target_domain))

    print(best_hyperparameters)
    # Plot the learning curve produced on the best trial of the model selection
    plot_learning_curves(results[idx]['train_errors'], results[idx]['val_errors'])
    plot_accuracies(results[idx]['train_accuracies'], results[idx]['val_accuracies'], show=True)