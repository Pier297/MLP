from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import generate_hyperparameters_random, gen_range
from MLP.Utils import argmin, generate_seed, average, change_seed
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

    # --- First Grid Search ---

    hyperparameters1 = {
        'loss_function_name'     : "MSE",
        'optimizer'              : "SGD", #optimizer = "SGD",
        'in_dimension'           : in_dimension,
        'out_dimension'          : out_dimension,
        'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
        'mini_batch_percentage'  : 1,
        'max_unlucky_epochs'     : 400,
        'max_epochs'             : 1000,
        'number_trials'          : 1,
        'validation_type'        : {'method': 'kfold', 'k': 5}, # validation_type={'method': 'holdout'},
        'target_domain'          : None,
        'lr'                     : [6e-3], # 0.6
        'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
        'l2'                     : [0],
        'momentum'               : [0.6],
        'adam_decay_rate_1'      : [0.9],
        'adam_decay_rate_2'      : [0.999],
        'hidden_layers'          : [([('tanh',20), ('tanh',20)],'linear')],
        'print_stats'            : False
    }
    hyperparameters1_stream = generate_hyperparameters(hyperparameters1)

    print(f'First grid search over: {len(hyperparameters1_stream)} configurations.')
    before_grid_search_time1                = time.perf_counter()
    best_hyperconfiguration1, best_results1 = grid_search(hyperparameters1_stream, training)
    after_grid_search_time1                 = time.perf_counter()

    # --- Refine the second Grid Search using a second Random Search ---

    #generations = 10
    #hyperparameters2 = {**best_hyperconfiguration1,
    #    'lr':       gen_range(best_hyperconfiguration1['lr'],       hyperparameters1['lr'],       method='uniform'),
    #    #'momentum': gen_range(best_hyperconfiguration1['momentum'], hyperparameters1['momentum'], method='uniform')
    #}
    #hyperparameters2_stream = generate_hyperparameters_random(hyperparameters2, generations)

    #print(f'First grid search over: {generations} configurations.')
    #before_grid_search_time2                = time.perf_counter()
    #best_hyperconfiguration2, best_results2 = grid_search(hyperparameters2_stream, training)
    #after_grid_search_time2                 = time.perf_counter()

    best_results2 = best_results1
    best_hyperconfiguration2 = best_hyperconfiguration1

    # --- Retraining with the final model ---

    training_epochs = ceil(average(list(map(lambda x: x["best_epoch"], best_results2["trials"]))))

    final_hyperparameters = {**best_hyperconfiguration2,
                             "max_epochs": training_epochs,
                             'seed': generate_seed(),
                             'print_stats': True}

    model = Sequential(final_hyperparameters)

    final_results = gradient_descent(model, training, test, final_hyperparameters)

    # --- Predicting and plotting ---

    train_output = predict(model, train_input)
    test_output  = predict(model, test_input)

    print("\n")
    print()
    print(f'Final model seed                     = {final_hyperparameters["seed"]}')
    print(f'Hyperparameters searched (1)         = {len(hyperparameters1)}')
    #print(f'Hyperparameters searched (2)         = {len(hyperparameters2)}')
    print(f'Best grid search validation epoch    = {training_epochs + 1} epochs')
    print(f'Best grid search validation error    = (MSE) {best_results2["val_error"]}')
    print(f'Final retrained MEE on training      = (MEE) {mean_euclidean_error(train_output, train_target)}')
    # CAREFUL! UNCOMMENT ONLY AT THE END OF THE ENTIRE EXPERIMENT
    print(f'Final retrained MEE on test          = (MEE) {mean_euclidean_error(test_output, test_target)}')
    print(f'Grid search total time (s) (1)      = {(after_grid_search_time1 - before_grid_search_time1)} seconds')
    #print(f'Grid search total time (s) (2)      = {(after_grid_search_time2 - before_grid_search_time2)} seconds')

    print("\nFinal hyperparameters\n\n", final_hyperparameters)

    loss_func_name = final_hyperparameters['loss_function_name']

    # Plot the weights and gradient norm during the final training
    plot_weights_norms(final_results['weights_norms'],   title=f'Weights norm during final training\n({time.asctime()})',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    plot_gradient_norms(final_results['gradient_norms'], title=f'Gradient norm during final training\n({time.asctime()})', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    # Plot the learning curves during the training of the best hyperparameter conf.
    plot_model_selection_learning_curves(best_results2['trials'], name=f"{loss_func_name} during Model Selection\n({time.asctime()})", highlight_best=True, file_name=f'MLP/cup/plots/model_selection_errors.png')

    # Plot the final learning curve while training on all the data
    plot_final_training_with_test_error(final_results['train_errors'],final_results['watch_errors'],name=loss_func_name, file_name=f'MLP/cup/plots/final_errors.png',
                                        skip_first_elements=0)

    plot_compare_outputs(train_output, train_target, name=f'Final training output comparison\n({time.asctime()})', file_name='MLP/cup/plots/scatter_train.png')

    plot_compare_outputs(test_output, test_target, name=f'Final test output comparison\n({time.asctime()})', file_name='MLP/cup/plots/scatter_test.png')

    end_plotting()
