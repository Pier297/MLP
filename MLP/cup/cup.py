from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.Utils import argmin, generate_seed, average
from MLP.cup.load_cup import load_cup
from multiprocessing import cpu_count
import numpy as np
import time
import random
import os
from math import ceil

global_seed = 2
random.seed(global_seed)
np.random.seed(global_seed)

np.seterr(all='raise')

if __name__ == '__main__':

    try:
        os.remove('MLP/cup/plots/')
    except:
        pass
    try:
        os.mkdir(f'MLP/cup/plots/')
    except:
        pass

    # Get training and internal test set
    (training, test) = load_cup()

    (in_dimension, out_dimension) = (10, 2)

    train_target = training[:, in_dimension:]
    train_input  = training[:, :in_dimension]

    test_target  = test[:, in_dimension:]
    test_input   = test[:, :in_dimension]

    # Start the grid search
    hyperparameters = generate_hyperparameters(
        loss_function_name     = "MSE",
        optimizer              = "SGD", #optimizer = "SGD",
        in_dimension           = in_dimension,
        out_dimension          = out_dimension,
        validation_percentage  = 0.20, # percentage of data into validation, remaining into training
        mini_batch_percentage  = 1,
        max_unlucky_epochs     = 20,
        max_epochs             = 300,
        number_trials          = 1,
        validation_type        = {'method': 'holdout'},#{'method': 'kfold', 'k': 5}, # validation_type={'method': 'holdout'},
        target_domain          = None,
        lr                     = [0.01], # 0.6
        lr_decay               = None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
        l2                     = [0.0001],
        momentum               = [0],
        adam_decay_rate_1      = [0.99],
        adam_decay_rate_2      = [0.999],
        hidden_layers          = [([('tanh',100), ('tanh', 100)],'linear')],
        print_stats            = False
    )

    # Run the grid-search which returns a list of dictionaries each containing
    # the best validation error found by early-stopping, the epochs where the val. error was the lowest and
    # an history of the training and validation errors during the training.

    before_grid_search_time = time.perf_counter()

    best_hyperparameters, best_results = grid_search(hyperparameters, training, cpu_count())

    after_grid_search_time = time.perf_counter()

    # Best trial
    best_trial = argmin(lambda t: t['best_val_error'], best_results['trials'])

    # Take the average between the best epochs of each trial
    final_training_epochs = ceil(average(list(map(lambda x: x["best_epoch"], best_results["trials"]))))

    # --- Retraining: define a new model with the best conf. and train on all the data ---

    final_hyperparameters = {**best_hyperparameters,
                             'max_epochs':  final_training_epochs,
                             'seed':        generate_seed(),
                             'print_stats': True}

    model = Sequential(final_hyperparameters)

    final_results = gradient_descent(model, training, None, final_hyperparameters, watching=test)

    # assert False
    # CAREFUL: we start dealing here with the test set!

    train_output = predict(model, train_input)
    test_output  = predict(model, test_input)

    print("\n")
    print()
    print(f'Final model seed                     = {final_hyperparameters["seed"]}')
    print(f'Hyperparameters searched             = {len(hyperparameters)}')
    print(f'Best grid search validation epoch    = {final_training_epochs + 1} epochs')
    print(f'Best grid search validation error    = (MSE) {best_results["val_error"]} ')
    print(f'Final retrained MEE on training      = (MEE) {mean_euclidean_error(train_output, train_target)}')
    # CAREFUL! UNCOMMENT ONLY AT THE END OF THE ENTIRE EXPERIMENT
    print(f'Final retrained MEE on test          = (MEE) {mean_euclidean_error(test_output, test_target)}')
    print(f'Grid search total time (s)           = {after_grid_search_time - before_grid_search_time} seconds')

    print("\nFinal hyperparameters\n\n", final_hyperparameters)

    # Plot the weights and gradient norm during the final training
    plot_weights_norms(final_results['weights_norms'],   title='Weights norm during final training',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    plot_gradient_norms(final_results['gradient_norms'], title='Gradient norm during final training', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    # Plot the learning curves during the training of the best hyperparameter conf.
    plot_model_selection_learning_curves(best_results['trials'], name=best_hyperparameters['loss_function_name'], highlight_best=True, file_name=f'MLP/cup/plots/model_selection_errors.png')

    # Plot the final learning curve while training on all the data
    plot_final_training_with_test_error(final_results['train_errors'],final_results['watch_errors'],name=best_hyperparameters['loss_function_name'], file_name=f'MLP/cup/plots/final_errors.png',
                                        skip_first_elements=0)

    plot_compare_outputs(train_output, train_target, name='Final training output comparison', file_name='MLP/cup/plots/scatter_train.png')

    plot_compare_outputs(test_output, test_target, name='Final test output comparison', file_name='MLP/cup/plots/scatter_test.png')

    end_plotting()
