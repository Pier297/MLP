from MLP.Network import Sequential
from MLP.Optimizers import Gradient_descent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE, accuracy
from MLP.GridSearch import generate_hyperparameters
from MLP.experiments.utils import load_monk, set_seed, split_train_set
from multiprocessing import Pool, cpu_count
from MLP.Regularizers import early_stopping
from MLP.ActivationFunctions import Tanh, Sigmoid
from itertools import repeat
from math import inf

set_seed(2)

def run_grid_search(args):
    conf, (train_X, train_Y, val_X, val_Y, activations_functions_names, target_domain) = args
    hidden_layers_activations = [Tanh() if func_str == 'tanh' else Sigmoid() for func_str in activations_functions_names]
    # Train the net on 'conf' and return (best_val_error, best_epoch)
    dimension_in = train_X[0].shape[0]
    dimension_out = train_Y[0].shape[0]

    val_errors = []
    epochs = []

    # How many times to repeat the training with the same configuration in order to reduce the validation error variance
    K = 3
    best_config = {'val_error': inf, 'epochs': 0, 'train_errors': [], 'val_errors': [], 'train_accuracies': [], 'val_accuracies': []}
    for t in range(K):
        model = Sequential(conf, dimension_in, dimension_out, hidden_layers_activations)

        if conf["loss_function"] == "MSE":
            loss_func = MSE()
        elif conf["loss_function"] == "Cross Entropy":
            loss_func = CrossEntropy()

        (train_errors, train_accuracies, val_errors, val_accuracies, early_best_epoch) = early_stopping(model, loss_func, conf["lr"], conf["l2"], conf["momentum"], conf["BATCH_SIZE"], train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS=50, MAX_EPOCHS=500, target_domain=target_domain)
        current_val_error = val_errors[early_best_epoch-1]
        if current_val_error < best_config['val_error']:
            best_config = {'val_error': current_val_error, 'epochs': early_best_epoch, 'train_errors': train_errors, 'val_errors': val_errors, 'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
    
    return best_config


if __name__ == '__main__':
    target_domain=(0, 1)
    hidden_layers_activations = ['tanh', 'sigmoid']

    (training_data_X, training_data_Y, test_X, test_Y) = load_monk(1, target_domain)
    (train_X, train_Y, val_X, val_Y) = split_train_set(training_data_X, training_data_Y, 0.8)

    hyperparameters = generate_hyperparameters(
        loss_func_values = ["Cross Entropy"],
        lr_values = [0.4],
        l2_values = [0],
        momentum_values = [0],
        hidden_layers_values = [[4]],
        BATCH_SIZE_values = [train_X.shape[0]]
    )

    with Pool(processes=cpu_count() - 1) as pool:
        results = pool.map(run_grid_search, zip(hyperparameters, repeat((train_X, train_Y, val_X, val_Y, hidden_layers_activations, target_domain))))

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
    
    plot_learning_curves(results[idx]['train_errors'], results[idx]['val_errors'])
    plot_accuracies(results[idx]['train_accuracies'], results[idx]['val_accuracies'], show=True)

    # --- Define a new model with the best conf. and train on all the data ---
    hidden_layers_activations = [Tanh() if func_str == 'tanh' else Sigmoid() for func_str in hidden_layers_activations]
    model = Sequential(best_hyperparameters, in_dimension=17, out_dimension=1, hidden_layers_activations=hidden_layers_activations)

    if best_hyperparameters['loss_function'] == 'MSE':
        loss_func = MSE()
    elif best_hyperparameters['loss_function'] == 'Cross Entropy':
        loss_func = CrossEntropy()


    (train_errors, val_errors, train_accuracies, val_accuracies) = Gradient_descent(model, training_data_X, training_data_Y, val_X, val_Y, loss_func, lr=best_hyperparameters["lr"], l2=best_hyperparameters["l2"], momentum=best_hyperparameters["momentum"], BATCH_SIZE=best_hyperparameters["BATCH_SIZE"], MAX_EPOCHS=best_epochs, target_domain=target_domain)

    print("Train accuracy =", accuracy(model, training_data_X, training_data_Y, target_domain))
    print("Validation accuracy =", accuracy(model, val_X, val_Y, target_domain))
    #print("Test accuracy =", accuracy(model, test_X, test_Y, target_domain))

    print(best_hyperparameters)