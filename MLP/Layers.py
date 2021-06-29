import numpy as np
from MLP.ActivationFunctions import Tanh, Sigmoid
from math import sqrt

def Dense(in_dimension: int, out_dimension: int, use_bias=True, activation_func=Tanh()):
    layer = {}
    init = sqrt(6/(out_dimension + in_dimension))
    layer["W"] = np.random.uniform(low=-init/2, high=init/2, size=(out_dimension, in_dimension))
    #layer["W"] = np.random.rand(out_dimension, in_dimension) * 0.3

    layer["b"] = np.zeros((out_dimension,))
    layer["use_bias"] = use_bias
    layer["in_dimension"] = in_dimension
    layer["out_dimension"] = out_dimension
    (layer["activation_func"], layer["activation_func_derivative"]) = activation_func
    return layer

def forward(layer, x):
    return layer["activation_func"](net(layer, x))

def net(layer, x):
    return np.dot(x, layer["W"].T) + layer["b"]