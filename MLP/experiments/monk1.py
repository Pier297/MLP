from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.Optimizers import GradientDescent
from MLP.Metrics import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE
from MLP.ModelSelection import GridSearch, generate_hyperparameters

from MLP.experiments.utils import load_monk, set_seed, split_train_set

seed = 42

set_seed(seed)

(train_X, train_Y, test_X, test_Y) = load_monk(1, target_domain=(-1, 1))

(train_X, train_Y, val_X, val_Y) = split_train_set(train_X, train_Y, 0.8)

(best_hyperparameters, train_errors, val_errors, train_accuracies, val_accuracies) = GridSearch(generate_hyperparameters(
    loss_func_values = ["MSE"],
    lr_values = [0.1, 0.3, 0.5],
    L2_values = [0],
    momentum_values = [0, 0.2, 0.8],
    hidden_layers_values = [[3]],
    BATCH_SIZE_values = [len(train_X)//2]
), in_dimension=17, out_dimension=1).search(train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS = 25, MAX_EPOCHS = 250, seed=seed)


print(best_hyperparameters)

plot_learning_curves(train_errors, val_errors)
plot_accuracies(train_accuracies, val_accuracies, show=True)


# --- Define a new model with the best conf. and train on all the data ---
model = Sequential()
model.from_configuration(best_hyperparameters, in_dimension=17, out_dimension=1)

if best_hyperparameters['loss_function'] == 'MSE':
    loss_func = MSE(best_hyperparameters["L2"])
elif best_hyperparameters['loss_function'] == 'Cross Entropy':
    loss_func = CrossEntropy(L2 = best_hyperparameters["L2"])
optimizer = GradientDescent(loss_function=loss_func, lr=best_hyperparameters["lr"], momentum=best_hyperparameters["momentum"], BATCH_SIZE=best_hyperparameters["BATCH_SIZE"])

(train_errors, train_accuracies, val_errors, val_accuracies) = optimizer.optimize(model, train_X, train_Y, val_X, val_Y, MAX_EPOCHS=best_hyperparameters["epochs"])

model.evaluate(test_X, test_Y)

plot_learning_curves(train_errors, val_errors)
plot_accuracies(train_accuracies, val_accuracies, show=True)