from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import accuracy, mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.Utils import argmin, generate_seed
from MLP.cup.load_cup import load_cup
from multiprocessing import cpu_count
import numpy as np
import time
import random
import os

global_seed = 1
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



    (training, test) = load_cup()

        # ---
        #lr_values = [0.1, 0.2, 0.4, 0.6],
        #l2_values = [0, 1e-5, 1e-3],
        #momentum_values = [0, 0.1, 0.2, 0.6, 0.9],
        #hidden_layers_values = [([('tanh',3)],'sigmoid'), ([('tanh',4)],'sigmoid')]
        # ---

    hyperparameters = generate_hyperparameters(
        loss_function_name     = "MSE",
        optimizer              = "SGD", #optimizer = "SGD",
        in_dimension           = 10,
        out_dimension          = 2,
        validation_percentage  = 0.95, # percentage of data into validation, remaining into training
        mini_batch_percentage  = 1,
        max_unlucky_epochs     = 10,
        max_epochs             = 1000,
        number_trials          = 1,
        validation_type        = {'method': 'holdout'},#{'method': 'kfold', 'k': 5}, # validation_type={'method': 'holdout'},
        target_domain          = None,
        lr                     = [0.01], # 0.6
        lr_decay               = None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
        l2                     = [1e-4],
        momentum               = [0],
        adam_decay_rate_1      = [0.99],
        adam_decay_rate_2      = [0.999],
        hidden_layers          = [([('tanh',50), ('tanh', 50)],'linear')],
        print_stats            = False
    )

    # seed 2
    # {'in_dimension': 17, 'out_dimension': 1, 'target_domain': (0, 1), 'validation_type': {'method': 'kfold', 'k': 5},
    # 'tion_nametion': 'Cross Entropy', 'lr': 0.4, 'l2': 0.001, 'momentum': 0.6, 'hidden_layers': ([('tanh', 3)], 'sigmoid'),
    # 'mini_batch_percentage': 0.8, 'mini_batch_percentage': 1}
    # fa crashare
    # {'in_dimension': 17, 'out_dimension': 1, 'target_domain': (0, 1), 'validation_type': {'method': 'kfold', 'k': 5},
    # 'loss_function': 'Cross Entropy', 'lr': 0.6, 'l2': 0.001, 'momentum': 0.9, 'hidden_layers': ([('tanh', 3)], 'sigmoid'),
    # 'mini_batch_percentage': 0.8, 'mini_batch_percentage': 1}

    # Run the grid-search which returns a list of dictionaries each containing
    # the best validation error found by early-stopping, the epochs where the val. error was the lowest and
    # an history of the training and validation errors during the training.

    before_grid_search_time = time.perf_counter()

    best_hyperparameters, best_results = grid_search(hyperparameters, training, cpu_count())

    after_grid_search_time = time.perf_counter()

    best_i = argmin(lambda t: t['best_val_error'], best_results['trials'])

    best_epoch = best_results["trials"][best_i]["best_epoch"]

    # --- Retraining: define a new model with the best conf. and train on all the data ---
    final_hyperparameters = {**best_hyperparameters,
                                'max_epochs':      best_epoch,
                                #'min_train_error': best_results["trials"][best_i]["train_errors"][best_epoch],
                                'seed':            generate_seed(),
                                'print_stats':     True}

    model = Sequential(final_hyperparameters)

    final_results = gradient_descent(model, training, None, final_hyperparameters, watching=test)

    train_input  = training[:, :model["in_dimension"]]
    train_target = training[:, model["in_dimension"]:]
    test_input   = test[:, :model["in_dimension"]]
    test_target  = test[:, model["in_dimension"]:]

    train_output = predict(model, train_input)
    test_output  = predict(model, test_input)

    print("\n")
    print(f'Final model seed                     = {final_hyperparameters["seed"]}')
    print(f'Hyperparameters searched             = {len(hyperparameters)}')
    print(f'Best grid search validation epoch    = {best_epoch + 1}')
    print(f'Best grid search validation error    = {best_results["trials"][best_i]["val_errors"][best_epoch]}')
    print(f'Final retrained MEE on training      = {mean_euclidean_error(train_output, train_target)}')
    # CAREFUL! UNCOMMENT ONLY AT THE END OF THE ENTIRE EXPERIMENT
    print(f'Final retrained MEE on test      (!) = {mean_euclidean_error(test_output, test_target)}')
    print(f'Grid search total time (s)           = {after_grid_search_time - before_grid_search_time} seconds')

    print("\nFinal hyperparameters\n\n", final_hyperparameters)

    # Plot the weights and gradient norm during the best trial of the best hyperparameter
    plot_weights_norms(best_results["trials"][best_i]["weights_norms"],   title='Weights norm during model selection',  file_name=f'MLP/cup/plots/model_selection_weights_norms.png')
    plot_gradient_norms(best_results["trials"][best_i]["gradient_norms"], title='Gradient norm during model selection', file_name=f'MLP/cup/plots/model_selection_gradient_norms.png')

    # Plot the weights and gradient norm during the final training
    plot_weights_norms(final_results['weights_norms'],   title='Weights norm during final training',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    plot_gradient_norms(final_results['gradient_norms'], title='Gradient norm during final training', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    # Plot the learning curves during the training of the best hyperparameter conf.
    plot_model_selection_learning_curves(best_results['trials'], name=best_hyperparameters['loss_function_name'], highlight_best=True, file_name=f'MLP/cup/plots/model_selection_errors.png')

    # Plot the final learning curve while training on all the data
    plot_final_training_with_test_error(final_results['train_errors'],final_results['watch_errors'],name=best_hyperparameters['loss_function_name'], file_name=f'MLP/cup/plots/final_errors.png',
               skip_first_elements=20)

    plot_compare_outputs(train_output, test_output, name='Output comparisons')

    end_plotting()