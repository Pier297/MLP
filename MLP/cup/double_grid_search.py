from MLP.Network import Sequential, predict
from MLP.GradientDescent import gradient_descent
from MLP.Plotting import *
from MLP.LossFunctions import mean_euclidean_error
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.RandomSearch import generate_hyperparameters_random, gen_range
from MLP.Utils import *
from MLP.cup.load_cup import load_blind_cup, load_cup
from MLP.cup.cup_hyperparameters import *
from MLP.cup.cup_hyperparameters import *
import numpy as np
from math import ceil

# double_grid_search.py

# Define the two helper functions to run the two separate grid searches.
# Each function combines the grid and the random search used for the two learning methods.

def best_k_grid_search_adam(n_training, hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0):

    # Start the first grid search

    _, best_results, validation_results = grid_search(hyperparameters_stream, n_training)

    for v in validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    # Sort them according to the validation error, and pick the first k models

    sorted_validations_results = sorted(enumerate(validation_results), key=lambda x: x[1]['val_error'])
    best_models_results = sorted_validations_results[:k]

    # Obtain the models configurations from the results

    (top_models_confs, _) = list(map(lambda x: hyperparameters_stream[x[0]], best_models_results)), list(map(lambda x: x[1], best_models_results))

    final_models_confs = []
    final_validation_results = []
    if n_random_search > 0:

        # Start the random search perturbations for each of the k models selected by the search

        for best_model_conf in top_models_confs:

            # Finally, start the random search by exploiting again the grid search function

            perturbated_best_model_conf, best_results, validation_results = \
                grid_search(generate_hyperparameters_random(best_model_conf, {**best_model_conf,
                                'lr':                gen_range(best_model_conf['lr'],                hyperparameters_space['lr'],                method='uniform', boundaries=(1e-9, inf)),
                                'l2':                gen_range(best_model_conf['l2'],                hyperparameters_space['l2'],                method='uniform', boundaries=(1e-9, inf)),
                                'adam_decay_rate_1': gen_range(best_model_conf['adam_decay_rate_1'], hyperparameters_space['adam_decay_rate_1'], method='uniform', boundaries=(0, 0.99999)),
                                'adam_decay_rate_2': gen_range(best_model_conf['adam_decay_rate_2'], hyperparameters_space['adam_decay_rate_2'], method='uniform', boundaries=(0, 0.99999))
                            }, n_random_search), n_training)

            # Collect the results

            final_models_confs.append(perturbated_best_model_conf)
            final_validation_results.append(best_results)

    for v in final_validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    return final_models_confs, final_validation_results

def best_k_grid_search_sgd(n_training, hyperparameters_stream, hyperparameters_space, k=1, n_random_search=0):

    # Start the first grid search

    _, best_results, validation_results = grid_search(hyperparameters_stream, n_training)

    for v in validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    # Sort them according to the validation error, and pick the first k models

    sorted_validations_results = sorted(enumerate(validation_results), key=lambda x: x[1]['val_error'])
    best_models_results = sorted_validations_results[:k]

    # Obtain the models configurations from the results

    (top_models_confs, _) = list(map(lambda x: hyperparameters_stream[x[0]], best_models_results)), list(map(lambda x: x[1], best_models_results))

    final_models_confs = []
    final_validation_results = []
    if n_random_search > 0:

        # Start the random search perturbations for each of the k models selected by the search

        for best_model_conf in top_models_confs:

            # Finally, start the random search by exploiting again the grid search function

            perturbated_best_model_conf, best_results, validation_results = \
                grid_search(generate_hyperparameters_random(best_model_conf, {**best_model_conf,
                                'lr':       gen_range(best_model_conf['lr'],       hyperparameters_space['lr'],       method='uniform', boundaries=(1e-9, inf)),
                                'l2':       gen_range(best_model_conf['l2'],       hyperparameters_space['l2'],       method='uniform', boundaries=(1e-9, inf)),
                                'momentum': gen_range(best_model_conf['momentum'], hyperparameters_space['momentum'], method='uniform', boundaries=(0, 0.99999))
                            }, n_random_search), n_training)

            # Collect the results

            final_models_confs.append(perturbated_best_model_conf)
            final_validation_results.append(best_results)

    for v in final_validation_results:
        v['optimizer_name'] = hyperparameters_space['optimizer']

    return final_models_confs, final_validation_results
