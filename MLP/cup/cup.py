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

global_seed = 0
random.seed(global_seed)
np.random.seed(global_seed)

def best_k_grid_search(hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0):
    print(f'First grid search over: {len(hyperparameters_stream)} configurations.')
    before_grid_search_time1                = time.perf_counter()
    best_hyperconfiguration, best_results, validation_results = grid_search(hyperparameters_stream, n_training)
    after_grid_search_time1                 = time.perf_counter()
    print(f'Finished first grid search in {after_grid_search_time1 - before_grid_search_time1} seconds')
    if n_random_search > 0:
        hyperparameters = {**best_hyperconfiguration,
            'lr':       gen_range(best_hyperconfiguration['lr'],   hyperparameters_space['lr'],    method='uniform'),
            #'momentum': gen_range(best_hyperconfiguration1['momentum'], hyperparameters1['momentum'], method='uniform')
        }
        hyperparameters_stream = generate_hyperparameters_random(hyperparameters, k)

        print(f'Second grid search over: {k} configurations.')
        before_grid_search_time2                = time.perf_counter()
        best_hyperconfiguration, best_results, validation_results = grid_search(hyperparameters_stream, n_training)
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
    n_models_ensemble = 4
    top_models_confs, top_validation_results = best_k_grid_search(hyperparameters1_stream, adam_hyperparameters, k=n_models_ensemble, n_random_search=10)

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
                             'print_stats': False} # TODO: Set to true..

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

    print("Final configurations")
    print(final_confs)

    print("Modelse MEE")
    print(models_mee)

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
    plot_weights_norms(final_results['weights_norms'],   title=f'Weights norm during final training\n({time.asctime()})',  file_name=f'MLP/cup/plots/final_weights_norms.png')
    plot_gradient_norms(final_results['gradient_norms'], title=f'Gradient norm during final training\n({time.asctime()})', file_name=f'MLP/cup/plots/final_gradient_norms.png')

    for i_ensemble, (grid_search_results, retraining_result) in enumerate(zip(top_validation_results, ensemble_results)):
        # Plot grid search

        # Plot the k curves of the validation
        if adam_hyperparameters['validation_type']['method'] == 'kfold':
            print("Length: ", len(grid_search_results['best_trial_plots']))
            assert len(grid_search_results['best_trial_plots']) == adam_hyperparameters['validation_type']['k']
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], name='Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')
        else:
            pass
            #get_trial_plots = lambda results: list(map(lambda r: r['plots'][0], results))
            #plot_model_selection_learning_curves(get_trial_plots(grid_search_results['trials']), name='Mean Squared Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')

        # Plot the final retraining
        plot_final_training_with_test_error(retraining_result['train_errors'], retraining_result['watch_errors'], name='Mean Squared Error', file_name=f'MLP/cup/plots/ensemble{i_ensemble}_retraining_errors.svg')


    # Plot the final learning curve while training on all the data
    #plot_final_training_with_test_error(final_results['train_errors'],final_results['watch_errors'],name=loss_func_name, file_name=f'MLP/cup/plots/final_errors.png',
    #                                    skip_first_elements=0)
    #plot_compare_outputs(train_output, train_target, name=f'Final training output comparison\n({time.asctime()})', file_name='MLP/cup/plots/scatter_train.png')
    #plot_compare_outputs(test_output, test_target, name=f'Final test output comparison\n({time.asctime()})', file_name='MLP/cup/plots/scatter_test.png')

    end_plotting()
