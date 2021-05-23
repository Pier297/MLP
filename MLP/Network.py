from MLP.Metrics import accuracy
from MLP.Layers import Dense

# Defines a Feed-forward neural network and provides
# a 'fit' method which is then used by the Model Selector.
class Sequential:
    def __init__(self):
        self.layers = []

    def from_configuration(self, configuration, in_dimension, out_dimension):
        self.add(Dense(in_dimension, configuration["hidden_layers"][0]))
        for i in range(len(configuration["hidden_layers"]) - 1):
            self.add(Dense(configuration["hidden_layers"][i], configuration["hidden_layers"][i + 1]))
        self.add(Dense(configuration["hidden_layers"][-1], out_dimension))

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # Returns (train_errors, train_accuracies, val_errors, val_accuracies)
    # TODO: Remove me or change me to either early stopping or model selection..
    def fit(self, train_X, train_Y, val_X, val_Y, optimizer):
        return optimizer.optimize(self, train_X, train_Y, val_X, val_Y)

    def evaluate(self, test_X, test_Y):
        print(f'\nTest accuracy = {accuracy(self, test_X, test_Y)}')

    # Returns a copy of itself with randomly initialized parameters.
    def reset(self):
        model = Sequential()
        for layer in self.layers:
            model.add(Dense(layer.dimension_in, layer.dimension_out, layer.use_bias, layer.activation_func_name))
        return model