import random
import numpy as np

# Dense layer
# It creates a fully connected layer with 'dimension_in' x 'dimension_out'
# number of weights.
class Dense:
    def __init__(self, dimension_in: int, dimension_out: int):
        self.W = np.random.rand(dimension_out, dimension_in)

    def forward(self, x):
        return np.dot(self.W, x)
    
    def backward(self):
        pass