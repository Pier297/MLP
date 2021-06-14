from MLP.Network import Sequential
from MLP.Optimizers import Gradient_descent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE, accuracy, loss_function_from_name
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.experiments.utils import load_monk, set_seed, split_train_set
from multiprocessing import cpu_count
from MLP.Regularizers import early_stopping
from MLP.ActivationFunctions import Tanh, Sigmoid
import numpy as np
from math import inf

set_seed(2)

if __name__ == '__main__':
    target_domain=(0, 1)

    (training, test) = load_monk(1, target_domain)

    hyperparameters = generate_hyperparameters(
        loss_func = "Cross Entropy",
        hidden_layers_activations = ['tanh', 'sigmoid'],
        in_dimension    = 17,
        out_dimension   = 1,
        target_domain = target_domain,
        train_percentage=0.8, # percentage of data into training, remaining into validation
        mini_batch_percentage=1,
        validation_type={'method': 'holdout'},
        #validation_type={'method': 'kfold', 'k': 5},
        # ---
        lr_values = [0.4, 0.6, 0.8],
        l2_values = [0],
        momentum_values = [0, 0.1, 0.2, 0.5, 0.7],
        hidden_layers_values = [[4]]
    )

    # Run the grid-search which returns a list of dictionaries each containing
    # the best validation error found by early-stopping, the epochs where the val. error was the lowest and
    # an history of the training and validation errors during the training.
    results = grid_search(hyperparameters, training, cpu_count())
    # Find the best hyperparameters configuration
    best_val_error = inf
    best_epochs = 0
    idx = -1
    # Search for the hyperparameter conf. with the lowest validation error
    for i, config in enumerate(results):
        val_error = config['val_error']
        if val_error < best_val_error:
            best_val_error = val_error
            best_epochs = config['epochs']
            idx = i

    best_hyperparameters = hyperparameters[idx]

    # --- Retraining: define a new model with the best conf. and train on all the data ---

    model = Sequential(best_hyperparameters)
    (train_errors, train_accuracies, val_errors, val_accuracies) = Gradient_descent(model, training, np.array([]), loss_function=loss_function_from_name(best_hyperparameters['loss_function']), lr=best_hyperparameters["lr"], l2=best_hyperparameters["l2"], momentum=best_hyperparameters["momentum"], train_percentage=best_hyperparameters["train_percentage"], MAX_EPOCHS=best_epochs, target_domain=target_domain)

    print("\nTrain accuracy =", accuracy(model, training, target_domain))
    #print("Test accuracy =", accuracy(model, test, target_domain))

    print("\n", best_hyperparameters)
    # Plot the learning curve produced on the best trial of the model selection
    plot_learning_curves(results[idx]['train_errors'], results[idx]['val_errors'], early_stopping_epoch=results[idx]['epochs'])
    plot_accuracies(results[idx]['train_accuracies'], results[idx]['val_accuracies'], early_stopping_epoch=results[idx]['epochs'], show=True)