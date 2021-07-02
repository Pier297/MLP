from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import generate_hyperparameters_random, gen_range
from MLP.Utils import *
from MLP.cup.load_cup import load_cup
import numpy as np
import time
import random
import os
from math import ceil

adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 1,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : [0.3],
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 2000,
    'number_trials'          : 1,
    'n_random_search'        : 30,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [1e-3, 5e-4, 5e-5], # 0.6
    'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
    'l2'                     : [1e-5],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.9],
    'adam_decay_rate_2'      : [0.999],
    'hidden_layers'          : [([('tanh',32), ('tanh', 32),],'linear'),],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False
}

global_seed = ceil((2**16 - 1) * np.random.rand())
random.seed(global_seed)
np.random.seed(global_seed)

def best_k_grid_search(hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0, training_data=[]):
    print(f'First grid search over: {len(hyperparameters_stream)} configurations.')
    before_grid_search_time1                = time.perf_counter()
    best_hyperconfiguration, best_results, validation_results = grid_search(hyperparameters_stream, training_data)
    after_grid_search_time1                 = time.perf_counter()
    print(f'Finished first grid search in {after_grid_search_time1 - before_grid_search_time1} seconds')
    if n_random_search > 0:
        hyperparameters = {**best_hyperconfiguration,
            'lr':       gen_range(best_hyperconfiguration['lr'],   hyperparameters_space['lr'],    method='uniform', boundaries=(1e-9, inf)),
            'momentum': gen_range(best_hyperconfiguration['momentum'], hyperparameters_space['momentum'], method='uniform', boundaries=(0, 0.99999))
        }
        hyperparameters_stream = generate_hyperparameters_random(hyperparameters, n_random_search)

        print(f'Second grid search over: {n_random_search} configurations.')
        before_grid_search_time2                = time.perf_counter()
        best_hyperconfiguration, best_results, validation_results = grid_search(hyperparameters_stream, training_data)
        after_grid_search_time2                 = time.perf_counter()
        print(f'Finished first grid search in {after_grid_search_time2 - before_grid_search_time2} seconds')

    # Get best k results from the grid search
    sorted_validations_results = sorted(enumerate(validation_results), key=lambda x: x[1]['val_error'])
    best_models_results = sorted_validations_results[:k]
    # hyperparameters_best_models, validation_results
    return list(map(lambda x: hyperparameters_stream[x[0]], best_models_results)), list(map(lambda x: x[1], best_models_results))

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

    first_training = training[:, :11]
    second_training = np.concatenate((training[:, :10], training[:, 11:]), axis=1)

    n_models_ensemble = 8

    (in_dimension, out_dimension) = (10, 1)

    train_input  = training[:, :in_dimension]
    train_target = training[:, 10:]
    train_first_target = training[:, 10]
    train_second_target = training[:, 11]

    test_input   = test[:, :in_dimension]
    test_target = test[:, 10:]
    test_first_target = test[:, 10]
    test_second_target = test[:, 11]

    training_statistics = data_statistics(training)

    n_training = normalize(training, training_statistics)
    n_test     = normalize(test,     training_statistics)

    n_first_training = n_training[:, :11]
    n_second_training = n_training[:, [0,1,2,3,4,5,6,7,8,9,11]]
    n_first_test = n_test[:, :11]
    n_second_test = n_test[:, [0,1,2,3,4,5,6,7,8,9,11]]

    n_train_input  = normalize(training[:, :in_dimension], training_statistics[:in_dimension])
    n_train_first_target = normalize_column(training[:, 10], training_statistics[10])
    n_train_second_target = normalize_column(training[:, 11], training_statistics[11])

    n_test_input  = normalize(test[:, :in_dimension], training_statistics[:in_dimension])
    n_test_first_target = normalize_column(test[:, 10], training_statistics[10])
    n_test_second_target = normalize_column(test[:, 11], training_statistics[11])

    # GRID SEARCH FOR FIRST MODEL

    hyperparameters1_stream = generate_hyperparameters(adam_hyperparameters)
    n_random_search = adam_hyperparameters['n_random_search']
    assert n_random_search >= 4
    top_models_confs1, top_validation_results1 = best_k_grid_search(hyperparameters1_stream, adam_hyperparameters, k=8,
                                                                  n_random_search=n_random_search, training_data=first_training)

    # GRID SEARCH FOR SECOND MODEL

    hyperparameters1_stream = generate_hyperparameters(adam_hyperparameters)
    n_random_search = adam_hyperparameters['n_random_search']
    assert n_random_search >= 4
    top_models_confs2, top_validation_results2 = best_k_grid_search(hyperparameters1_stream, adam_hyperparameters, k=8,
                                                                  n_random_search=n_random_search, training_data=second_training)

    
    # Train the best k models
    ensemble_train_outputs = np.zeros((n_train_input.shape[0], 2))
    ensemble_test_outputs  = np.zeros((n_test_input.shape[0], 2))
    ensemble_results = []
    final_confs = []
    models_mee = []
    for best_hyperconf1, results1, best_hyperconf2, results2 in zip(top_models_confs1, top_validation_results1, top_models_confs2, top_validation_results2):
        # Train model for first output
        retraining_epochs = results1["epochs"]

        final_hyperparameters = {**best_hyperconf1,
                             "max_epochs": retraining_epochs,
                             'seed': generate_seed(),
                             'print_stats': False}

        final_confs.append(final_hyperparameters)

        model_first_output = Sequential(final_hyperparameters)

        final_results = gradient_descent(model_first_output, n_first_training, None, final_hyperparameters, watching=n_first_test)
        #ensemble_results.append(final_results)
        
        # Train model for second output
        retraining_epochs = results2["epochs"]

        final_hyperparameters = {**best_hyperconf2,
                             "max_epochs": retraining_epochs,
                             'seed': generate_seed(),
                             'print_stats': False}

        final_confs.append(final_hyperparameters)

        model_second_output = Sequential(final_hyperparameters)

        final_results = gradient_descent(model_second_output, n_second_training, None, final_hyperparameters, watching=n_second_test)

        # --- Combine the 2 outputs ---

        train_first_output = denormalize_column(predict(model_first_output, n_train_input), training_statistics[10])
        test_first_output  = denormalize_column(predict(model_first_output, n_test_input),  training_statistics[10])
        train_second_output = denormalize_column(predict(model_second_output, n_train_input), training_statistics[11])
        test_second_output  = denormalize_column(predict(model_second_output, n_test_input),  training_statistics[11])

        train_output = np.concatenate((train_first_output, train_second_output), axis=1)
        test_output = np.concatenate((test_first_output, test_second_output), axis=1)

        train_mee = mean_euclidean_error(train_output, train_target)
        test_mee  = mean_euclidean_error(test_output, test_target)
        models_mee.append((train_mee, test_mee))

        ensemble_train_outputs += train_output
        ensemble_test_outputs  += test_output

    # Combine the ensemble outputs into a single prediction by avg the outputs
    ensemble_train_outputs /= n_models_ensemble
    ensemble_test_outputs  /= n_models_ensemble

    #print("Final configurations")
    #print(final_confs)

    #print("Models MEE")
    #print(models_mee)

    print("\n")
    print()

    ensemble_train_mee = mean_euclidean_error(ensemble_train_outputs, train_target)
    ensemble_test_mee  = mean_euclidean_error(ensemble_test_outputs,  test_target)

    print(f'Final retrained MEE on training      = (MEE)       {ensemble_train_mee}')
    # CAREFUL! UNCOMMENT ONLY AT THE END OF THE ENTIRE EXPERIMENT
    print(f'Final retrained MEE on test          = (MEE)       {ensemble_test_mee}')

    with open(f'MLP/cup/results/ensemble_mee.txt', 'w') as f:
        f.write("\nFinal configurations\n")
        f.write(str(final_confs))
        f.write("\nModels MEE:\n")
        f.write('\n'.join([str(train_mee) + ' ' + str(test_mee) for train_mee, test_mee in models_mee]))
        f.write('\n\nEnsemble Train MEE: ')
        f.write(str(ensemble_train_mee))
        f.write('\nEnsemble TEST MEE: ')
        f.write(str(ensemble_test_mee))
        f.write('\n')


    loss_func_name = final_hyperparameters['loss_function_name']

    # Plot the weights and gradient norm during the final training
    #plot_weights_norms(final_results['weights_norms'],   title=f'Weights norm during final training',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    #plot_gradient_norms(final_results['gradient_norms'], title=f'Gradient norm during final training', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    """ for i_ensemble, (grid_search_results, retraining_result) in enumerate(zip(top_validation_results, ensemble_results)):
        # Plot the k curves of the validation
        if adam_hyperparameters['validation_type']['method'] == 'kfold':
            print("Length: ", len(grid_search_results['best_trial_plots']))
            assert len(grid_search_results['best_trial_plots']) == adam_hyperparameters['validation_type']['k']
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], name=f'Ensemble {i_ensemble}: Grid Search Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')
        else:
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], name=f'Ensemble {i_ensemble}: Grid Search Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')

        # Plot the final retraining
        plot_final_training_with_test_error(retraining_result['train_errors'], retraining_result['watch_errors'], name=f'Ensemble {i_ensemble}: Final Training Mean Squared Error', file_name=f'MLP/cup/plots/ensemble{i_ensemble}_retraining_errors.svg')
 """

    plot_compare_outputs(ensemble_train_outputs, train_target, name=f'Final training output comparison', file_name='MLP/cup/plots/scatter_train.svg')
    plot_compare_outputs(ensemble_test_outputs, test_target, name=f'Final test output comparison', file_name='MLP/cup/plots/scatter_test.svg')

    end_plotting()