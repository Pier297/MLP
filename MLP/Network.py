
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

    def fit(self, X, Y, lr = 0.01, MAX_EPOCHS = 100):
        pass