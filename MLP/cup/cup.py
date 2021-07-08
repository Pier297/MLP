from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import generate_hyperparameters_random, gen_range
from MLP.Utils import *
from MLP.cup.load_cup import load_blind_cup, load_cup
from MLP.cup.cup_hyperparameters import *
from MLP.cup.double_grid_search import *
import numpy as np
import random
import os
from math import ceil

# cup.py
# Main code used for obtaining the CUP results.

# Set the global seed either to a random value of a set one.
global_seed = ceil((2**16 - 1) * np.random.rand())
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

    # Get training and internal test set.

    (training, test) = load_cup()

    # Select the total number of ensemble models

    n_models_ensemble = 8

    # Dimensions used for the CUP

    (in_dimension, out_dimension) = (10, 2)

    # Training data

    train_input  = training[:, :in_dimension]
    train_target = training[:, in_dimension:]

    # Test data (!)

    test_input   = test[:, :in_dimension]
    test_target  = test[:, in_dimension:]

    # Compute the training statistics on the training dataset

    training_statistics = data_statistics(training)

    # Apply standardization to all involved datasets:

    n_training = normalize(training, training_statistics)
    n_test     = normalize(test,     training_statistics)

    n_train_input  = normalize(training[:, :in_dimension], training_statistics[:in_dimension])
    n_train_target = normalize(training[:, in_dimension:], training_statistics[in_dimension:])

    # Load the blind cup input data

    cup_data = load_blind_cup()
    cup_inputs = cup_data[:, 1:]

    n_cup_input   = normalize(cup_inputs[:, :in_dimension], training_statistics[:in_dimension])
    n_test_input  = normalize(test      [:, :in_dimension], training_statistics[:in_dimension])
    n_test_target = normalize(test      [:, in_dimension:], training_statistics[in_dimension:])

    #################################### Grid searches ##############################################

    # Run the grid search + random search for the first ensemble trained with Adam

    hyperparameters1_stream = generate_hyperparameters(adam_hyperparameters)
    n_random_search = adam_hyperparameters['n_random_search']
    top_models_confs, top_validation_results = best_k_grid_search_adam(n_training, hyperparameters1_stream, adam_hyperparameters, k=n_models_ensemble//2, n_random_search=n_random_search)

    # Run the grid search + random search for the first ensemble trained with SGD

    hyperparameters2_stream = generate_hyperparameters(sgd_hyperparameters)
    n_random_search = sgd_hyperparameters['n_random_search']
    top_models_confs2, top_validation_results2 = best_k_grid_search_sgd(n_training, hyperparameters2_stream, sgd_hyperparameters, k=n_models_ensemble//2, n_random_search=n_random_search)

    top_models_confs       += top_models_confs2
    top_validation_results += top_validation_results2

    # Retrain the best k models by collecting the outputs

    ensemble_train_outputs = np.zeros((n_train_input.shape[0], adam_hyperparameters['out_dimension']))
    ensemble_test_outputs  = np.zeros((n_test_input.shape[0], adam_hyperparameters['out_dimension']))
    ensemble_cup_outputs   = np.zeros((n_cup_input.shape[0], adam_hyperparameters['out_dimension']))

    # Final results and output for each ensemble

    ensemble_results = []
    final_confs      = []
    models_mee       = []
    gridsearch_mee   = []

    #################################### Retraining ##############################################

    # We calculate the outputs for each model of the ensemble and save the retraining results.

    for ensemble_i, (best_hyperconf, results) in enumerate(zip(top_models_confs, top_validation_results)):

        # Compute the final hyperparameters

        final_hyperparameters = {**best_hyperconf,
                                 "max_epochs": results["epochs"],
                                 'seed': generate_seed(),
                                 'print_stats': False}

        final_confs.append(final_hyperparameters)

        # Retrain the final model with the correct number of epochs

        model = Sequential(final_hyperparameters)
        final_results = gradient_descent(model, n_training, None, final_hyperparameters, watching=n_test)
        ensemble_results.append(final_results)

        # Output the model results, and denormalize them

        train_output = denormalize(predict(model, n_train_input), training_statistics[in_dimension:])
        test_output  = denormalize(predict(model, n_test_input),  training_statistics[in_dimension:])
        cup_output   = denormalize(predict(model, n_cup_input),   training_statistics[in_dimension:])

        # Compute the relevant statistics and collect the outputs

        train_mee = mean_euclidean_error(train_output, train_target)
        test_mee  = mean_euclidean_error(test_output, test_target)

        models_mee.append((train_mee, test_mee))
        gridsearch_mee.append((results['metric_val_error'],
                               results['metric_val_error_var'],
                               results['metric_train_error'],
                               results['metric_train_error_var']))

        ensemble_train_outputs += train_output
        ensemble_test_outputs  += test_output
        ensemble_cup_outputs   += cup_output

    # Combine the ensemble outputs into a single prediction by averaging the outputs

    ensemble_train_outputs /= n_models_ensemble
    ensemble_test_outputs  /= n_models_ensemble
    ensemble_cup_outputs   /= n_models_ensemble

    ensemble_train_mee = mean_euclidean_error(ensemble_train_outputs, train_target)
    ensemble_test_mee  = mean_euclidean_error(ensemble_test_outputs,  test_target)

    # Save the blind cup results on the relevant file

    with open(f'MLP/cup/results/lambda00_ML-CUP20-TS.csv', 'w') as f:
        for i, row in enumerate(ensemble_cup_outputs):
            f.write(str(i+1))
            f.write(',')
            f.write(str(row[0]))
            f.write(',')
            f.write(str(row[1]))
            f.write('\n')

    # Save the final results on the final ensemble_mee.txt file

    with open(f'MLP/cup/results/ensemble_mee.txt', 'w') as f:
        f.write('Global seed ' + str(global_seed))
        f.write("\nFinal configurations\n")
        for conf in final_confs:
            f.write('\n' + str(conf))
        f.write("\nModels MEE (train, val, test):\n")
        f.write('\n'.join([str(train_mee) + ' ' + str(test_mee) for train_mee, test_mee in models_mee]))
        f.write(repr(gridsearch_mee))
        f.write('\n\nEnsemble Train MEE: ')
        f.write(str(ensemble_train_mee))
        f.write('\nEnsemble TEST MEE: ')
        f.write(str(ensemble_test_mee))
        f.write('\n')

    #################################### Plotting ##############################################

    # Produce the plots for each ensembled model

    for i_ensemble, (grid_search_results, retraining_result) in enumerate(zip(top_validation_results, ensemble_results)):

        # Plot the k curves of the validation

        if adam_hyperparameters['validation_type']['method'] == 'kfold' and sgd_hyperparameters['validation_type']['method'] == 'kfold':
            # In the case of k-fold, plot all the folds on the same graph
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], metric=True, name=f'Ensemble {i_ensemble + 1} ({"SGD" if grid_search_results["optimizer_name"] == "SGDN" else "Adam"}): Grid Search Mean Euclidean Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')
        else:
            plot_model_selection_learning_curves(grid_search_results['best_trial_plots'], metric=True, name=f'Ensemble {i_ensemble + 1} ({"SGD" if grid_search_results["optimizer_name"] == "SGDN" else "Adam"}): Grid Search Mean Euclidean Error', highlight_best=True, file_name=f'MLP/cup/plots/ensemble{i_ensemble}_model_selection_errors.svg')

        # Plot the final retraining results

        plot_final_training_with_test_error(retraining_result['metric_train_errors'], retraining_result['metric_watch_errors'], name=f'Ensemble {i_ensemble + 1} ({"SGD" if i_ensemble > 3 else "Adam"}): Final Retraining Mean Euclidean Error', file_name=f'MLP/cup/plots/ensemble{i_ensemble}_retraining_errors.svg')

    # Display the final scatter plot outputs

    plot_compare_outputs(ensemble_train_outputs, train_target, name=f'Final training output comparison', file_name='MLP/cup/plots/scatter_train.svg')
    plot_compare_outputs(ensemble_test_outputs, test_target, name=f'Final test output comparison', file_name='MLP/cup/plots/scatter_test.svg')
    plot_compare_outputs(ensemble_cup_outputs, None, name=f'Blind outputs', file_name='MLP/cup/plots/scatter_cup.svg')

    end_plotting()
