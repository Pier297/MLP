from MLP.Network import Sequential
from MLP.Optimizers import Gradient_descent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE, accuracy
from MLP.GridSearch import generate_hyperparameters
from MLP.experiments.utils import load_monk, set_seed, split_train_set
from multiprocessing import Pool, cpu_count
from MLP.Regularizers import early_stopping
from MLP.ActivationFunctions import Tanh, Sigmoid
from math import inf

set_seed(1)

target_domain=(0, 1)
hidden_layers_activations = [Tanh(), Sigmoid()]

(train_X, train_Y, test_X, test_Y) = load_monk(1, target_domain)
(train_X, train_Y, val_X, val_Y) = split_train_set(train_X, train_Y, 0.8)


def run_grid_search(conf, train_X=train_X, train_Y=train_Y, val_X=val_X, val_Y=val_Y, hidden_layers_activations=hidden_layers_activations):
    # Train the net on 'conf' and return (best_val_error, best_epoch)
    dimension_in = train_X[0].shape[0]
    dimension_out = train_Y[0].shape[0]

    val_errors = []
    epochs = []

    # How many times to repeat the training with the same configuration in order to reduce the validation error variance
    K = 3
    for t in range(K):
        model = Sequential(conf, dimension_in, dimension_out, hidden_layers_activations)

        if conf["loss_function"] == "MSE":
            loss_func = MSE()
        elif conf["loss_function"] == "Cross Entropy":
            loss_func = CrossEntropy()

        (train_errors, val_errors, early_best_epoch) = early_stopping(model, loss_func, conf["lr"], conf["l2"], conf["momentum"], conf["BATCH_SIZE"], train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS=50, MAX_EPOCHS=500)
        val_errors.append(val_errors[early_best_epoch-1])
        epochs.append(early_best_epoch)
    return sum(val_errors)/K, sum(epochs) // K


if __name__ == '__main__':
    hyperparameters = generate_hyperparameters(
        loss_func_values = ["Cross Entropy"],
        lr_values = [0.4, 0.6, 0.8],
        l2_values = [0],
        momentum_values = [0],
        hidden_layers_values = [[3], [4]],
        BATCH_SIZE_values = [train_X.shape[0]]
    )

    pool = Pool(processes=cpu_count())
    results = pool.map(run_grid_search, hyperparameters)

    best_val_error = inf
    best_epochs = 0
    idx = -1
    # Search for the hyperparameter conf. with the lowest validation error
    for i, (val_error, epochs) in enumerate(results):
        if val_error < best_val_error:
            best_val_error = val_error
            best_epochs = epochs
            idx = i

    best_hyperparameters = hyperparameters[idx]

    # --- Define a new model with the best conf. and train on all the data ---
    model = Sequential(best_hyperparameters, in_dimension=17, out_dimension=1, hidden_layers_activations=hidden_layers_activations)

    if best_hyperparameters['loss_function'] == 'MSE':
        loss_func = MSE()
    elif best_hyperparameters['loss_function'] == 'Cross Entropy':
        loss_func = CrossEntropy()

    (train_errors, val_errors) = Gradient_descent(model, train_X, train_Y, val_X, val_Y, loss_func, lr=best_hyperparameters["lr"], l2=best_hyperparameters["l2"], momentum=best_hyperparameters["momentum"], BATCH_SIZE=best_hyperparameters["BATCH_SIZE"], MAX_EPOCHS=best_epochs)

    print("Train accuracy =", accuracy(model, train_X, train_Y, target_domain))
    print("Validation accuracy =", accuracy(model, val_X, val_Y, target_domain))
    #print("Test accuracy =", accuracy(model, test_X, test_Y, target_domain))

    print(best_hyperparameters)

    plot_learning_curves(train_errors, val_errors, show=True, name='Training on train+val sets')