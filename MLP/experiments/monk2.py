from MLP.Network import Sequential
from MLP.Optimizers import Gradient_descent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import accuracy, loss_function_from_name
from MLP.GridSearch import generate_hyperparameters, grid_search
from MLP.experiments.utils import load_monk, set_seed, argmin
from multiprocessing import cpu_count
import numpy as np

set_seed(2)

if __name__ == '__main__':
    target_domain=(0, 1)

    (training, test) = load_monk(2, target_domain)

    hyperparameters = generate_hyperparameters(
        loss_func = "Cross Entropy",
        in_dimension    = 17,
        out_dimension   = 1,
        target_domain = target_domain,
        train_percentage=0.8, # percentage of data into training, remaining into validation
        mini_batch_percentage=1,
        #validation_type={'method': 'holdout'},
        validation_type={'method': 'kfold', 'k': 5},
        # ---
        lr_values = [0.1, 0.2, 0.4, 0.6],
        l2_values = [0, 1e-5, 1e-3],
        momentum_values = [0, 0.1, 0.2, 0.6, 0.9],
        hidden_layers_values = [([('tanh',3)],'sigmoid'), ([('tanh',4)],'sigmoid')]
    )

    # seed 2
    # {'in_dimension': 17, 'out_dimension': 1, 'target_domain': (0, 1), 'validation_type': {'method': 'kfold', 'k': 5},
    # 'loss_function': 'Cross Entropy', 'lr': 0.6, 'l2': 1e-05, 'momentum': 0.9, 'hidden_layers': ([('tanh', 4)], 'sigmoid'),
    # 'train_percentage': 0.8, 'mini_batch_percentage': 1}

    # Run the grid-search which returns a list of dictionaries each containing
    # the best validation error found by early-stopping, the epochs where the val. error was the lowest and
    # an history of the training and validation errors during the training.
    best_hyperparameters, best_results = grid_search(hyperparameters, training, cpu_count())

    best_i = argmin(lambda t: t['val_error'], best_results['trials'])
    
    
    # --- Retraining: define a new model with the best conf. and train on all the data ---

    model = Sequential(best_hyperparameters)
    (train_errors, train_accuracies, val_errors, val_accuracies) = Gradient_descent(model, training, np.array([]), loss_function=loss_function_from_name(best_hyperparameters['loss_function']), lr=best_hyperparameters["lr"], l2=best_hyperparameters["l2"], momentum=best_hyperparameters["momentum"], train_percentage=best_hyperparameters["train_percentage"], MAX_EPOCHS=best_results["trials"][best_i]["epochs"], target_domain=target_domain)

    print("\nTrain accuracy =", accuracy(model, training, target_domain))
    print("Test accuracy =", accuracy(model, test, target_domain))

    print("\n", best_hyperparameters)
    plot_learning_curves(best_results['trials'], name=best_hyperparameters['loss_function'], highlight_best=True, file_name='MLP/experiments/results/monk2/errors.png')
    plot_accuracies(best_results['trials'], highlight_best=True, show=True, file_name='MLP/experiments/results/monk2/accuracies.png')