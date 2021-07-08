import numpy as np
from MLP.ActivationFunctions import Tanh, Sigmoid
from math import sqrt

# Layers.py

# Definition for a fully connected layer of the Network.

def Dense(in_dimension: int, out_dimension: int, use_bias=True, activation_func=Tanh(), weights_init = {'method': 'linear', 'range': 0.7}):
    """
    Main constructor for a dense layer of the network.
    This simply initializes the network and returns its data object.

    :param in_dimension: input dimension of the layer
    :param out_dimension: output dimension of the layer
    :param use_bias: if the bias is needed for this layer
    :param activation_func: compound function representing the activation function and its derivative
    :param weights_init: dictionary representing the initialization mechanism
    :return: a dictionary representing the layer information, composed of
                W: the weights (nparray matrix)
                b: the bias (nparray)
                <the informations passed to the layer function>
    """

    layer = {}

    # Select the weight initialization method

    if weights_init['method'] == 'linear':
        layer["W"] = ((np.random.rand(out_dimension, in_dimension)-0.5)*2) * weights_init['range']
    if weights_init['method'] == 'gaussian':
        layer["W"] = np.random.randn(out_dimension, in_dimension) * weights_init['range']
    elif weights_init['method'] == 'normalized':

        # Normalized initialization

        init = sqrt(6/(out_dimension + in_dimension))
        layer["W"] = np.random.uniform(low=-init, high=init, size=(out_dimension, in_dimension))
    else:
        raise Exception(f'Invalid weights initialization {str(weights_init)}')

    # Unpack the values and save them inside the model dictionary

    layer["b"] = np.zeros((out_dimension,))
    layer["use_bias"] = use_bias
    layer["in_dimension"] = in_dimension
    layer["out_dimension"] = out_dimension
    (layer["activation_func"], layer["activation_func_derivative"]) = activation_func

    return layer

def forward(layer, x):
    """
    Compute the forward step of a single layer, from the input matrix to the output one.

    :param layer: layer structure of the network
    :param x: input matrix
    :return: output matrix
    """
    return layer["activation_func"](net(layer, x))

def net(layer, x):
    """
    Compute the collective network output of a single layer, without activation function.

    :param layer: layer structure of the network
    :param x: input matrix
    :return: output matrix
    """
    return np.dot(x, layer["W"].T) + layer["b"]
