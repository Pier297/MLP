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

totalt=0

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

    for monk_id, name, monk_hyperparameters in monks:
        monktimestart = time.perf_counter()

        try:
            os.mkdir(f'MLP/monk/plots/monk{name}/')
        except:
            pass

        target_domain = (0, 1)

        monk_hyperparameters_stream = generate_hyperparameters(monk_hyperparameters)

        (training, test) = load_monk(monk_id, target_domain)

        print(f'First grid search over: {len(monk_hyperparameters_stream)} configurations.')
        before_grid_search_time                = time.perf_counter()

        best_hyperconfiguration1, best_results1, _ = grid_search(monk_hyperparameters_stream, training)

        # --- Refine the second Grid Search using a second Random Search ---

        """ generations = 100
        hyperparameters2 = {**best_hyperconfiguration1,
            'lr':       gen_range(best_hyperconfiguration1['lr'],       monk_hyperparameters['lr'],       method='uniform'),
            'momentum': gen_range(best_hyperconfiguration1['momentum'], monk_hyperparameters['momentum'], method='uniform')
        }

        hyperparameters2_stream = generate_hyperparameters_random(hyperparameters2, generations)

        print(f'Second grid search over: {generations} configurations.')
        best_hyperconfiguration2, best_results2, _ = grid_search(hyperparameters2_stream, training)

        after_grid_search_time                 = time.perf_counter()"""

        best_results2 = best_results1
        best_hyperconfiguration2 = best_hyperconfiguration1

        # --- Retraining with the final model ---
        retraining_epochs = best_results2["epochs"]

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

        mse_func = MSE()

        train_error = loss_func.eval(train_output, train_target)
        test_error  = loss_func.eval(test_output,  test_target)
        train_mse = mse_func.eval(train_output, train_target)
        test_mse = mse_func.eval(test_output, test_target)

        variance_train_error = loss_func.std(train_output, train_target)**2
        variance_test_error  = loss_func.std(test_output, test_target)**2

        train_accuracy = accuracy(train_output, train_target, target_domain)
        test_accuracy  = accuracy(test_output, test_target, target_domain)

        trials_train_errors     = best_results2["trials_best_train_errors"]
        trials_train_accuracies = best_results2["trials_best_train_accuracies"]
        trials_val_errors       = best_results2["trials_best_val_errors"]
        trials_val_accuracies   = best_results2["trials_best_val_accuracies"]

        avg_grid_search_train_errors     = np.average(trials_train_errors)
        avg_grid_search_train_accuracies = np.average(trials_train_accuracies)
        avg_grid_search_val_errors       = np.average(trials_val_errors)
        avg_grid_search_val_accuracies   = np.average(trials_val_accuracies)

        variance_trials_train_errors   = np.std(trials_train_errors)**2
        variance_trials_train_accuracy = np.std(trials_train_accuracies)**2
        variance_trials_val_errors     = np.std(trials_val_errors)**2
        variance_trials_val_accuracy   = np.std(trials_val_accuracies)**2

        monktimestop = time.perf_counter()

        print("Monk:", monk_id, monktimestop - monktimestart)
        totalt += monktimestop - monktimestart
    print(totalt)
