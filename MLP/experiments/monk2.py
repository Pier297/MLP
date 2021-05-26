from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.Optimizers import GradientDescent
from MLP.Plotting import plot_learning_curves, plot_accuracies
from MLP.LossFunctions import CrossEntropy, MSE
from MLP.GridSearch import GridSearch, generate_hyperparameters

from MLP.experiments.utils import load_monk, set_seed, split_train_set

set_seed(1)

(train_X, train_Y, test_X, test_Y) = load_monk(1, target_domain=(-1, 1))

(train_X, train_Y, val_X, val_Y) = split_train_set(train_X, train_Y, 0.8)

model = Sequential()
model.add(Dense(17, 5))
model.add(Dense(5, 1))

optimizer = GradientDescent(lr=0.5, momentum=0.3, BATCH_SIZE=train_X.shape[0], loss_function=MSE(), l2=0.0)

(train_errors, train_accuracies, val_errors, val_accuracies) = optimizer.optimize(model, train_X, train_Y,val_X, val_Y, MAX_EPOCHS=400)

#(train_errors, train_accuracies) = early_stopping(model, optimizer, *split_train_set(train_X, train_Y, 0.8), MAX_UNLUCKY_STEPS = 10)

model.evaluate(test_X, test_Y)

plot_learning_curves(train_errors)
plot_accuracies(train_accuracies, show=True)