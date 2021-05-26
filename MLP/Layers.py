import numpy as np
from MLP.ActivationFunctions import Tanh
from math import sqrt

# Dense layer
# It creates a fully connected layer with 'dimension_in' x 'dimension_out'
# number of weights.
class Dense:
    def __init__(self, dimension_in: int, dimension_out: int, use_bias=True, activation_func=Tanh(), sd: float = 0.3):
        #self.W = sd * np.random.randn(dimension_out, dimension_in)
        #normalized_init = sqrt(6/(dimension_out + dimension_in))
        #self.W = np.random.uniform(low=-normalized_init, high=normalized_init, size=(dimension_out, dimension_in))

        init = 0.7
        self.W = np.random.uniform(low=-init/2, high=init/2, size=(dimension_out, dimension_in))
        if use_bias:
            self.b = np.random.uniform(low=-init/2, high=init/2, size=(dimension_out, 1))
        else:
            self.b = np.zeros((dimension_out, 1))
        self.use_bias = use_bias
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out

        (self.activation_func, self.activation_func_derivative) = activation_func

    def forward(self, x):
        return self.activation_func(self.net(x))

    def net(self, x):
        return np.reshape(np.reshape(np.dot(self.W, x), (self.dimension_out, 1)) + self.b, (self.dimension_out, 1))