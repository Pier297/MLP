from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.Optimizers import GradientDescent
from MLP.Metrics import plot_learning_curves, plot_accuracies
from MLP.Regularizers import early_stopping
from MLP.LossFunctions import CrossEntropy, MSE

from MLP.experiments.utils import load_monk, set_seed, split_train_set

set_seed(1)

(train_X, train_Y, test_X, test_Y) = load_monk(1, target_domain=(0, 1))

model = Sequential()
model.add(Dense(17, 3, activation_func='sigmoid'))
model.add(Dense(3, 1, activation_func='sigmoid'))

#loss_func = MSE(L2 = 0.0001)
loss_func = CrossEntropy(L2 = 0)
optimizer = GradientDescent(loss_function=loss_func, lr=0.75, momentum=0.3, BATCH_SIZE=10)

(train_errors, train_accuracies) = optimizer.optimize(model, train_X, train_Y, MAX_EPOCHS=250)

#(train_errors, train_accuracies) = early_stopping(model, optimizer, *split_train_set(train_X, train_Y, 0.8), MAX_UNLUCKY_STEPS = 10)

model.evaluate(test_X, test_Y)

plot_learning_curves(train_errors)
plot_accuracies(train_accuracies, show=True)