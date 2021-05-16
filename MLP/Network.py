
# Defines a Feed-forward neural network and provides
# a 'fit' method which is then used by the Model Selector.
class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(self, train_X, train_Y, val_X, val_Y, optimizer):
        optimizer.optimize(self, train_X, train_Y, val_X, val_Y)

    def evaluate(self, test_X, test_Y):
        corrects = 0
        for i in range(test_X.shape[0]):
            if test_Y[i] == (1 if self.predict(test_X[i]) >= 0 else -1):
                corrects += 1
        print(f'\nTest accuracy = {corrects / test_X.shape[0]}')