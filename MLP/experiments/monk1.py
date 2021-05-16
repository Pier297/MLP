from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.Optimizers import GradientDescent

from MLP.experiments.utils import load_monk, set_seed, split_train_set

set_seed(1)

(train_X, train_Y, test_X, test_Y) = load_monk(3)

model = Sequential()
model.add(Dense(17, 15))
model.add(Dense(15, 15))
model.add(Dense(15, 1))

optimizer = GradientDescent(lr=1, MAX_EPOCHS=150, BACTH_SIZE=20, MAX_UNLUCKY_STEPS=1)

model.fit(*split_train_set(train_X, train_Y, 0.8), optimizer)

model.evaluate(test_X, test_Y)