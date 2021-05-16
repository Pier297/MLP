import numpy as np

# Dense layer
# It creates a fully connected layer with 'dimension_in' x 'dimension_out'
# number of weights.
class Dense:
    def __init__(self, dimension_in: int, dimension_out: int, use_bias=True, activation_func: str = 'tanh', sd: float = 0.3):
        self.W = sd * np.random.randn(dimension_out, dimension_in)
        self.b = np.zeros((dimension_out, 1))
        self.use_bias = use_bias
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out
        if activation_func == 'tanh':
            self.activation_func = np.tanh
            self.activation_func_derivative = lambda x: (1 - np.tanh(x)**2)

    def forward(self, x):
        return self.activation_func(self.net(x))

    def net(self, x):
        return np.reshape(np.reshape(np.dot(self.W, x), (self.dimension_out, 1)) + self.b, (self.dimension_out, 1))