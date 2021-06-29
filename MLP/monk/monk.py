from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import *
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import *
from MLP.Utils import *
from MLP.monk.load_monk import load_monk
from MLP.monk.monk_hyperparameters import *
from multiprocessing import cpu_count
import numpy as np
import time
import random
import os
from math import *

global_seed = int(2**16 * np.random.rand())
random.seed(global_seed)
np.random.seed(global_seed)

monks = [(1, '1',    monk1_hyperparameters),
         #(2, '2',    monk2_hyperparameters),
         #(3, '3',    monk3_monk_hyperparameters_stream),
         #(3, '3reg', monk3_hyperparameters_reg_stream),
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

    for monk_id, name, monk_hyperparameters in monks:
        try:
            os.mkdir(f'MLP/monk/plots/monk{name}/')
        except:
            pass

        target_domain = (0, 1)

        monk_hyperparameters_stream = generate_hyperparameters(monk_hyperparameters)

        (training, test) = load_monk(monk_id, target_domain)

        print(f'First grid search over: {len(monk_hyperparameters_stream)} configurations.')
        before_grid_search_time                = time.perf_counter()
        
        best_hyperconfiguration1, best_results1 = grid_search(monk_hyperparameters_stream, training)
        
        # --- Refine the second Grid Search using a second Random Search ---

        print("Hyperconfiguration chosen by the first grid search")
        print(best_hyperconfiguration1)

        generations = 30
        hyperparameters2 = {**best_hyperconfiguration1,
            'lr':       gen_range(best_hyperconfiguration1['lr'],       monk_hyperparameters['lr'],       method='uniform'),
            'momentum': gen_range(best_hyperconfiguration1['momentum'], monk_hyperparameters['momentum'], method='uniform')
        }

        hyperparameters2_stream = generate_hyperparameters_random(hyperparameters2, generations)

        print(f'Second grid search over: {generations} configurations.')
        best_hyperconfiguration2, best_results2 = grid_search(hyperparameters2_stream, training)

        after_grid_search_time                 = time.perf_counter()

        #best_results2 = best_results1
        #best_hyperconfiguration2 = best_hyperconfiguration1

        # --- Retraining with the final model ---

        retraining_epochs = ceil(average(list(map(lambda x: x["best_epoch"], best_results2["trials"]))))

        final_hyperparameters = {**best_hyperconfiguration2,
                                "max_epochs": retraining_epochs,
                                'seed': generate_seed()}

        model = Sequential(final_hyperparameters)

        final_results = gradient_descent(model, training, None, final_hyperparameters, watching=test)

        train_input  = training[:, :model["in_dimension"]]
        train_target = training[:, model["in_dimension"]:]
        test_input   = test[:, :model["in_dimension"]]
        test_target  = test[:, model["in_dimension"]:]

        train_output = predict(model, train_input)
        test_output  = predict(model, test_input)

        loss_func = loss_function_from_name(final_hyperparameters['loss_function_name'])
        
        train_error = loss_func.eval(train_output, train_target)
        test_error  = loss_func.eval(test_output,  test_target)

        std_train_error = loss_func.std(train_output, train_target)
        std_test_error  = loss_func.std(test_output, test_target)

        train_accuracy = accuracy(train_output, train_target, target_domain)
        test_accuracy  = accuracy(test_output, test_target, target_domain)

        trials_train_errors     = get_trials_best_epoch_property(best_results2["trials"], 'train_errors')
        trials_train_accuracies = get_trials_best_epoch_property(best_results2["trials"], 'train_accuracies')
        trials_val_errors       = get_trials_best_epoch_property(best_results2["trials"], 'val_errors')
        trials_val_accuracies   = get_trials_best_epoch_property(best_results2["trials"], 'val_accuracies')
        
        avg_grid_search_train_errors     = np.average(trials_train_errors)
        avg_grid_search_train_accuracies = np.average(trials_train_accuracies)
        avg_grid_search_val_errors       = np.average(trials_val_errors)
        avg_grid_search_val_accuracies   = np.average(trials_val_accuracies)
       
        std_trials_train_errors   = np.std(trials_train_errors)
        std_trials_train_accuracy = np.std(trials_train_accuracies)
        std_trials_val_errors     = np.std(trials_val_errors)
        std_trials_val_accuracy   = np.std(trials_val_accuracies)
 
        print("\n")
        print(f"Monk {name}")
        print(f'Final model seed                     = {final_hyperparameters["seed"]}')
        print(f'Hyperparameters searched             = {len(monk_hyperparameters_stream)}')
        print(f'Best grid search train error         = {avg_grid_search_train_errors}, std_trials={std_trials_train_errors}')
        print(f'Best grid search train accuracy      = {avg_grid_search_train_accuracies}, std_trials={std_trials_train_accuracy}')
        print(f'Best grid search validation error    = {avg_grid_search_val_errors}, std_trials={std_trials_val_errors}')
        print(f'Best grid search validation accuracy = {avg_grid_search_val_accuracies}, std_trials={std_trials_val_accuracy}')
        print(f'Final selected epoch                 = {retraining_epochs + 1}')
        print(f'Final selected train accuracy        = {train_accuracy}')
        print(f'Final selected train error           = {train_error}, std={std_train_error}')
        print(f'Final selected test error            = {test_error}, std={std_test_error}')
        print(f'Final test accuracy                  = {test_accuracy}')
        print(f'Grid search total time (s)           = {after_grid_search_time - before_grid_search_time} seconds')

        print("\nFinal hyperparameters\n\n", final_hyperparameters)

        # Plot the weights and gradient norm during the final training
        plot_weights_norms(final_results['weights_norms'],   title='Weights norm during final training',  file_name=f'MLP/monk/plots/monk{name}/final_weights_norms.svg')
        plot_gradient_norms(final_results['gradient_norms'], title='Gradient norm during final training', file_name=f'MLP/monk/plots/monk{name}/final_gradient_norms.svg')

        # Plot the learning curves during the training of the best hyperparameter conf.
        plot_model_selection_learning_curves(best_results2['trials'], name=final_hyperparameters['loss_function_name'], highlight_best=True, file_name=f'MLP/monk/plots/monk{name}/model_selection_errors.svg')
        plot_model_selection_accuracies(best_results2['trials'], highlight_best=True,                                                       file_name=f'MLP/monk/plots/monk{name}/model_selection_accuracies.svg')

        # Plot the final learning curve while training on all the data
        plot_final_training_with_test_error     (final_results['train_errors'],     final_results['watch_errors'],     name=final_hyperparameters['loss_function_name'], file_name=f'MLP/monk/plots/monk{name}/final_errors.svg')
        plot_final_training_with_test_accuracies(final_results['train_accuracies'], final_results['watch_accuracies'],                                                  file_name=f'MLP/monk/plots/monk{name}/final_accuracies.svg')

        # Print table of results to file
        with open(f'MLP/monk/results/monk{name}.txt', 'w') as f:
            f.write(str(final_hyperparameters))
            f.write(f"Monk {name}")
            f.write(f'\nGlobal seed = {global_seed}\n')
            f.write(f'\nFinal model seed                     = {final_hyperparameters["seed"]}')
            f.write(f'\nHyperparameters searched             = {len(monk_hyperparameters_stream)}')
            f.write(f'\nBest grid search train error         = {avg_grid_search_train_errors}, std_trials={std_trials_train_errors}')
            f.write(f'\nBest grid search train accuracy      = {avg_grid_search_train_accuracies}, std_trials={std_trials_train_accuracy}')
            f.write(f'\nBest grid search validation error    = {avg_grid_search_val_errors}, std_trials={std_trials_val_errors}')
            f.write(f'\nBest grid search validation accuracy = {avg_grid_search_val_accuracies}, std_trials={std_trials_val_accuracy}')
            f.write(f'\nFinal selected epoch                 = {retraining_epochs + 1}')
            f.write(f'\nFinal selected train accuracy        = {train_accuracy}')
            f.write(f'\nFinal selected train error           = {train_error}, std={std_train_error}')
            f.write(f'\nFinal selected test error            = {test_error}, std={std_test_error}')
            f.write(f'\nFinal test accuracy                  = {test_accuracy}')
            f.write(f'\nGrid search total time (s)           = {after_grid_search_time - before_grid_search_time} seconds')

        #end_plotting()