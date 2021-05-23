import numpy as np
from math import sqrt

# Dense layer
# It creates a fully connected layer with 'dimension_in' x 'dimension_out'
# number of weights.
class Dense:
    def __init__(self, dimension_in: int, dimension_out: int, use_bias=True, activation_func: str = 'tanh', sd: float = 0.3):
        #self.W = sd * np.random.randn(dimension_out, dimension_in)
        #normalized_init = sqrt(6/(dimension_out + dimension_in))
        #self.W = np.random.uniform(low=-normalized_init, high=normalized_init, size=(dimension_out, dimension_in))

        init = 1/sqrt(dimension_in)
        self.W = np.random.uniform(low=-init, high=init, size=(dimension_out, dimension_in))
        self.b = np.zeros((dimension_out, 1))
        self.use_bias = use_bias
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out
        self.activation_func_name = activation_func
        if activation_func == 'tanh':
            self.activation_func = np.tanh
            self.activation_func_derivative = lambda x: (1 - np.tanh(x)**2)
        elif activation_func == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
            self.activation_func_derivative = lambda x: self.activation_func(x) * (1 - self.activation_func(x))
        else:
            raise ValueError(f'{activation_func} not supported as a layer activation function.')

    def forward(self, x):
        return self.activation_func(self.net(x))

    def net(self, x):
        return np.reshape(np.reshape(np.dot(self.W, x), (self.dimension_out, 1)) + self.b, (self.dimension_out, 1))