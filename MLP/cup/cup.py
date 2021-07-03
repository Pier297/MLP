from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import generate_hyperparameters_random, gen_range
from MLP.Utils import *
from MLP.cup.load_cup import load_cup
from MLP.cup.cup_hyperparameters import *
import numpy as np
import time
import random
import os
from datetime import datetime
from math import ceil

global_seed = ceil((2**16 - 1) * np.random.rand())
random.seed(global_seed)
np.random.seed(global_seed)

def best_k_grid_search_adam(n_training, hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0):
    print(f'First grid search over: {len(hyperparameters_stream)} configurations.')
    before_grid_search_time1                = time.perf_counter()
    _, best_results, validation_results = grid_search(hyperparameters_stream, n_training)
    after_grid_search_time1                 = time.perf_counter()
    print(f'Finished first grid search in {after_grid_search_time1 - before_grid_search_time1} seconds')

    # Get best k results from the grid search
    for v in validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']
    sorted_validations_results = sorted(enumerate(validation_results), key=lambda x: x[1]['val_error'])
    best_models_results = sorted_validations_results[:k]

    (top_models_confs, top_validation_results) = list(map(lambda x: hyperparameters_stream[x[0]], best_models_results)), list(map(lambda x: x[1], best_models_results))

    final_models_confs = []
    final_validation_results = []
    if n_random_search > 0:
        print('\n-------- Start random searches --------\n')
        for i_random_search, best_model_conf in enumerate(top_models_confs):
            print(f'(START) Random {i_random_search} search over: {n_random_search} configurations.')
            before_grid_search_time2 = time.perf_counter()
            perturbated_best_model_conf, best_results, validation_results = \
                grid_search(generate_hyperparameters_random(best_model_conf, {**best_model_conf,
                                'lr':                gen_range(best_model_conf['lr'],                hyperparameters_space['lr'],                method='uniform', boundaries=(1e-9, inf)),
                                'adam_decay_rate_1': gen_range(best_model_conf['adam_decay_rate_1'], hyperparameters_space['adam_decay_rate_1'], method='uniform', boundaries=(0, 0.99999)),
                                'adam_decay_rate_2': gen_range(best_model_conf['adam_decay_rate_2'], hyperparameters_space['adam_decay_rate_2'], method='uniform', boundaries=(0, 0.99999))
                            }, n_random_search), n_training)
            after_grid_search_time2 = time.perf_counter()
            final_models_confs.append(perturbated_best_model_conf)
            final_validation_results.append(best_results)
            print(f'(END) Random {i_random_search} search in {after_grid_search_time2 - before_grid_search_time2} seconds')

    for v in final_validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    # hyperparameters_best_models, validation_results
    return final_models_confs, final_validation_results
def best_k_grid_search_sgd(n_training, hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0):
    print(f'First grid search over: {len(hyperparameters_stream)} configurations.')
    before_grid_search_time1                = time.perf_counter()
    _, best_results, validation_results = grid_search(hyperparameters_stream, n_training)
    after_grid_search_time1                 = time.perf_counter()
    print(f'Finished first grid search in {after_grid_search_time1 - before_grid_search_time1} seconds')

    # Get best k results from the grid search
    for v in validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']
    sorted_validations_results = sorted(enumerate(validation_results), key=lambda x: x[1]['val_error'])
    best_models_results = sorted_validations_results[:k]

    (top_models_confs, top_validation_results) = list(map(lambda x: hyperparameters_stream[x[0]], best_models_results)), list(map(lambda x: x[1], best_models_results))

    final_models_confs = []
    final_validation_results = []
    if n_random_search > 0:
        print('\n-------- Start random searches --------\n')
        for i_random_search, best_model_conf in enumerate(top_models_confs):
            print(f'(START) Random {i_random_search} search over: {n_random_search} configurations.')
            before_grid_search_time2 = time.perf_counter()
            perturbated_best_model_conf, best_results, validation_results = \
                grid_search(generate_hyperparameters_random(best_model_conf, {**best_model_conf,
                                'lr':       gen_range(best_model_conf['lr'],       hyperparameters_space['lr'],       method='uniform', boundaries=(1e-9, inf)),
                                'momentum': gen_range(best_model_conf['momentum'], hyperparameters_space['momentum'], method='uniform', boundaries=(0, 0.99999))
                            }, n_random_search), n_training)
            after_grid_search_time2 = time.perf_counter()
            final_models_confs.append(perturbated_best_model_conf)
            final_validation_results.append(best_results)
            print(f'(END) Random {i_random_search} search in {after_grid_search_time2 - before_grid_search_time2} seconds')

    for v in final_validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    # hyperparameters_best_models, validation_results
    return final_models_confs, final_validation_results

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

    n_models_ensemble = 8

    (in_dimension, out_dimension) = (10, 2)

    train_input  = training[:, :in_dimension]
    train_target = training[:, in_dimension:]

    test_input   = test[:, :in_dimension]
    test_target  = test[:, in_dimension:]

    training_statistics = data_statistics(training)

    n_training = normalize(training, training_statistics)
    n_test     = normalize(test,     training_statistics)

    n_train_input  = normalize(training[:, :in_dimension], training_statistics[:in_dimension])
    n_train_target = normalize(training[:, in_dimension:], training_statistics[in_dimension:])

    n_test_input  = normalize(test[:, :in_dimension], training_statistics[:in_dimension])
    n_test_target = normalize(test[:, in_dimension:], training_statistics[in_dimension:])

    # Run first grid search for the first ensemble trained with ADAM
    hyperparameters1_stream = generate_hyperparameters(adam_hyperparameters)
    n_random_search = adam_hyperparameters['n_random_search']
    assert n_random_search >= 4
    top_models_confs, top_validation_results = best_k_grid_search_adam(n_training, hyperparameters1_stream, adam_hyperparameters, k=n_models_ensemble//2, n_random_search=n_random_search)

    # Run second grid search for the second ensemble trained with SGD
    hyperparameters2_stream = generate_hyperparameters(sgd_hyperparameters)
    n_random_search = sgd_hyperparameters['n_random_search']
    assert n_random_search >= 4
    top_models_confs2, top_validation_results2 = best_k_grid_search_sgd(n_training, hyperparameters2_stream, sgd_hyperparameters, k=n_models_ensemble//2, n_random_search=n_random_search)

    top_models_confs += top_models_confs2
    top_validation_results += top_validation_results2

    # Train the best k models
    ensemble_train_outputs = np.zeros((n_train_input.shape[0], adam_hyperparameters['out_dimension']))
    ensemble_test_outputs  = np.zeros((n_test_input.shape[0], adam_hyperparameters['out_dimension']))
    ensemble_results = []
    final_confs = []
    models_mee = []
    for best_hyperconf, results in zip(top_models_confs, top_validation_results):
        retraining_epochs = results["epochs"]

        final_hyperparameters = {**best_hyperconf,
                             "max_epochs": retraining_epochs,
                             'seed': generate_seed(),
                             'print_stats': False}

        final_confs.append(final_hyperparameters)

        model = Sequential(final_hyperparameters)

        final_results = gradient_descent(model, n_training, None, final_hyperparameters, watching=n_test)
        ensemble_results.append(final_results)

        train_output = denormalize(predict(model, n_train_input), training_statistics[in_dimension:])
        test_output  = denormalize(predict(model, n_test_input),  training_statistics[in_dimension:])

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

    with open(f'MLP/cup/results/ensemble_mee_{ensemble_test_mee}.txt', 'w') as f:
        f.write('Global seed ' + str(global_seed))
        f.write("\nFinal configurations\n")
        for conf in final_confs:
            f.write('\n' + str(conf))
        f.write("\nModels MEE:\n")
        f.write('\n'.join([str(train_mee) + ' ' + str(test_mee) for train_mee, test_mee in models_mee]))
        f.write('\n\nEnsemble Train MEE: ')
        f.write(str(ensemble_train_mee))
        f.write('\nEnsemble TEST MEE: ')
        f.write(str(ensemble_test_mee))
        f.write('\n')


    loss_func_name = final_hyperparameters['loss_function_name']

    # Plot the weights and gradient norm during the final training
    plot_weights_norms(final_results['weights_norms'],   title=f'Weights norm during final training',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    plot_gradient_norms(final_results['gradient_norms'], title=f'Gradient norm during final training', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    for i_ensemble, (grid_search_results, retraining_result) in enumerate(zip(top_validation_results, ensemble_results)):
        # Plot the k curves of the validation
        if adam_hyperparameters['validation_type']['method'] == 'kfold':
            print("Length: ", len(grid_search_results['best_trial_plots']))
            assert len(grid_search_results['best_trial_plots']) == adam_hyperparameters['validation_type']['k']
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], name=f'Ensemble {i_ensemble} {grid_search_results["optimizer_name"]}: Grid Search Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')
        else:
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], name=f'Ensemble {i_ensemble} {grid_search_results["optimizer_name"]}: Grid Search Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')

        # Plot the final retraining
        plot_final_training_with_test_error(retraining_result['train_errors'], retraining_result['watch_errors'], name=f'Ensemble {i_ensemble} {grid_search_results["optimizer_name"]}: Final Training Mean Squared Error', file_name=f'MLP/cup/plots/ensemble{i_ensemble}_retraining_errors.svg')


    plot_compare_outputs(ensemble_train_outputs, train_target, name=f'Final training output comparison', file_name='MLP/cup/plots/scatter_train.svg')
    plot_compare_outputs(ensemble_test_outputs, test_target, name=f'Final test output comparison', file_name='MLP/cup/plots/scatter_test.svg')

    end_plotting()