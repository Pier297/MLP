from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import *
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import *
from MLP.Utils import *
from MLP.monk.load_monk import load_monk
from MLP.monk.monk_hyperparameters import *
import numpy as np
import time
import random
import os
from math import *

# monk.py
# Main code used for obtaining the MONK's results.

# Set the global seed either to a random value of a set one.
global_seed = int(2**16 * np.random.rand())
random.seed(global_seed)
np.random.seed(global_seed)

# Global list indicating the tasks and hyperparameters to be executed.
monks = [(1, '1',    monk1_hyperparameters),
         (2, '2',    monk2_hyperparameters),
         (3, '3',    monk3_hyperparameters),
         (3, '3reg', monk3_hyperparameters_reg),
         ]

if __name__ == '__main__':
    try:
        os.remove('MLP/monk/plots/')
    except:
        pass
    try:
        os.mkdir(f'MLP/monk/plots/')
    except:
        pass

    # For each desired dataset to explore:

    for monk_id, name, monk_hyperparameters in monks:
        try:
            os.mkdir(f'MLP/monk/plots/monk{name}/')
        except:
            pass

        target_domain = (0, 1)

        # Load the MONK dataset

        (training, test) = load_monk(monk_id, target_domain)

        # Generate the hyperparameters list from the general configuration

        monk_hyperparameters_stream = generate_hyperparameters(monk_hyperparameters)

        # Perform the grid search

        best_hyperconfiguration, best_results, _ = grid_search(monk_hyperparameters_stream, training)

        # Retrain on the entire training dataset

        retraining_epochs = best_results["epochs"]

        final_hyperparameters = {**best_hyperconfiguration,
                                "max_epochs": retraining_epochs,
                                'seed': generate_seed()}

        model = Sequential(final_hyperparameters)

        final_results = gradient_descent(model, training, None, final_hyperparameters, watching=test)

        # Finally, evaluate the model on the datasets; first, split them in input and outputs

        train_input  = training[:, :model["in_dimension"]]
        train_target = training[:, model["in_dimension"]:]
        test_input   = test[:, :model["in_dimension"]]
        test_target  = test[:, model["in_dimension"]:]

        # Use the model to predict train and test outputs

        train_output = predict(model, train_input)
        test_output  = predict(model, test_input)

        loss_func = CrossEntropy()
        mse_func = MSE()

        train_error = loss_func.eval(train_output, train_target)
        test_error  = loss_func.eval(test_output,  test_target)
        train_mse = mse_func.eval(train_output, train_target)
        test_mse = mse_func.eval(test_output, test_target)

        variance_train_error = loss_func.std(train_output, train_target)**2
        variance_test_error  = loss_func.std(test_output, test_target)**2

        train_accuracy = accuracy(train_output, train_target, target_domain)
        test_accuracy  = accuracy(test_output, test_target, target_domain)

        trials_train_errors     = best_results["trials_best_train_errors"]
        trials_train_accuracies = best_results["trials_best_train_accuracies"]
        trials_val_errors       = best_results["trials_best_val_errors"]
        trials_val_accuracies   = best_results["trials_best_val_accuracies"]

        avg_grid_search_train_errors     = np.average(trials_train_errors)
        avg_grid_search_train_accuracies = np.average(trials_train_accuracies)
        avg_grid_search_val_errors       = np.average(trials_val_errors)
        avg_grid_search_val_accuracies   = np.average(trials_val_accuracies)

        variance_trials_train_errors   = np.std(trials_train_errors)**2
        variance_trials_train_accuracy = np.std(trials_train_accuracies)**2
        variance_trials_val_errors     = np.std(trials_val_errors)**2
        variance_trials_val_accuracy   = np.std(trials_val_accuracies)**2

        # Print table of results to file

        with open(f'MLP/monk/results/monk{name}.txt', 'w') as f:
            f.write(f"Monk {name}")
            f.write(f'\nGlobal seed = {global_seed}\n')
            f.write(f'\nFinal model seed                     = {final_hyperparameters["seed"]}')
            f.write(f'\nHyperparameters searched             = {len(monk_hyperparameters_stream)}')
            f.write(f'\nBest grid search train error         = {avg_grid_search_train_errors}, sigma2 = {variance_trials_train_errors}')
            f.write(f'\nBest grid search train accuracy      = {avg_grid_search_train_accuracies}, sigma2 = {variance_trials_train_accuracy}')
            f.write(f'\nBest grid search validation error    = {avg_grid_search_val_errors}, sigma2 = {variance_trials_val_errors}')
            f.write(f'\nBest grid search validation accuracy = {avg_grid_search_val_accuracies}, sigma2 = {variance_trials_val_accuracy}')
            f.write(f'\nFinal selected epoch                 = {retraining_epochs + 1}')
            f.write(f'\nFinal selected train accuracy        = {train_accuracy}')
            f.write(f'\nFinal selected train error           = {train_error}, std={variance_train_error}')
            f.write(f'\nFinal selected test error            = {test_error}, std={variance_test_error}')
            f.write(f'\nFinal test accuracy                  = {test_accuracy}')
            f.write(f'\n')
            f.write(f'\nTrain MSE = {train_mse}')
            f.write(f'\nTest  MSE = {test_mse}')
            f.write("\nFinal hyperparameters\n")
            f.write(str(final_hyperparameters))

        #################################### Plotting ##############################################

        # Plot the learning curves during the training of the best hyperparameter configuration

        if monk_hyperparameters['validation_type']['method'] == 'kfold':
            # In the case of k-fold, plot all the folds on the same graph
            plot_model_selection_learning_curves(best_results['best_trial_plots'], name=final_hyperparameters['loss_function_name'], highlight_best=True, file_name=f'MLP/monk/plots/monk{name}/model_selection_errors.svg')
            plot_model_selection_accuracies(best_results['best_trial_plots'], highlight_best=True,                                                       file_name=f'MLP/monk/plots/monk{name}/model_selection_accuracies.svg')
        else:
            get_trial_plots = lambda results: list(map(lambda r: r['plots'][0], results))
            plot_model_selection_learning_curves(get_trial_plots(best_results['trials']), name=final_hyperparameters['loss_function_name'], highlight_best=True, file_name=f'MLP/monk/plots/monk{name}/model_selection_errors.svg')
            plot_model_selection_accuracies(get_trial_plots(best_results['trials']), highlight_best=True,                                                       file_name=f'MLP/monk/plots/monk{name}/model_selection_accuracies.svg')

        # Plot the final learning curve while training on all the data

        plot_final_training_with_test_error     (final_results['train_errors'],     final_results['watch_errors'],     name=final_hyperparameters['loss_function_name'], file_name=f'MLP/monk/plots/monk{name}/final_errors.svg')
        plot_final_training_with_test_accuracies(final_results['train_accuracies'], final_results['watch_accuracies'],                                                  file_name=f'MLP/monk/plots/monk{name}/final_accuracies.svg')

        end_plotting()
