from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.Optimizers import GradientDescent
from MLP.Metrics import plot_learning_curves, plot_accuracies
from MLP.Regularizers import early_stopping

from MLP.experiments.utils import load_monk, set_seed, split_train_set

set_seed(1)

(train_X, train_Y, test_X, test_Y) = load_monk(3)

model = Sequential()
model.add(Dense(17, 15))
model.add(Dense(15, 1))

optimizer = GradientDescent(lr=0.3, momentum=0.3, BATCH_SIZE=25)

#(train_errors, train_accuracies) = optimizer.optimize(model, train_X, train_Y, MAX_EPOCHS=200)

(train_errors, train_accuracies) = early_stopping(model, optimizer, *split_train_set(train_X, train_Y, 0.8), MAX_UNLUCKY_STEPS = 50, MAX_EPOCHS=250)

model.evaluate(test_X, test_Y)

plot_learning_curves(train_errors)
plot_accuracies(train_accuracies, show=True)