import numpy as np
from MLP.ActivationFunctions import Tanh
from math import sqrt

def Dense(dimension_in: int, dimension_out: int, use_bias=True, activation_func=Tanh()):
    layer = {}
    init = sqrt(6/(dimension_out + dimension_in))
    layer["W"] = np.random.uniform(low=-init/2, high=init/2, size=(dimension_out, dimension_in))
    
    if use_bias:
        layer["b"] = np.random.uniform(low=-init/2, high=init/2, size=(dimension_out, 1))
    else:
        layer["b"] = np.zeros((dimension_out, 1))
    layer["use_bias"] = use_bias
    layer["dimension_in"] = dimension_in
    layer["dimension_out"] = dimension_out

    (layer["activation_func"], layer["activation_func_derivative"]) = activation_func
    return layer


def forward(layer, x):
    return layer["activation_func"](net(layer, x))


def net(layer, x):
    return np.reshape(np.reshape(np.dot(layer["W"], x), (layer["dimension_out"], 1)) + layer["b"], (layer["dimension_out"], 1))