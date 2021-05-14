from MLP.Network import Sequential
from MLP.Layers import Dense
from MLP.experiments.utils import load_monk, set_seed

set_seed(1)

(train_X, train_Y, test_X, test_Y) = load_monk(1)

print(train_X.shape)
print(train_Y.shape)

model = Sequential()
model.add(Dense(17, 2))
model.add(Dense(2, 1))

print(model.predict(train_X[0]))

