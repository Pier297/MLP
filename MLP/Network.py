import numpy as np
from MLP.Layers import Dense, forward
from MLP.ActivationFunctions import activation_function_from_name
import random

# Network.py

# Main definitions for Sequential networks, which are the
# only kinds of networks implemented so far.
# Utilizes Dense layers as network components.

def Sequential(conf):
    """
    Main constructor for a sequentially connected whole network.
    The network will be composed of Layers, each with its bias and weight.

    :param conf: configuration dictionary with the network structure and settings
    :return: a dictionary object representing the network and its layers
    """

    in_dimension  = conf["in_dimension"]
    out_dimension = conf["out_dimension"]

    model = {}

    model["seed"]          = conf["seed"]
    model["layers"]        = []
    model["in_dimension"]  = in_dimension
    model["out_dimension"] = out_dimension

    np.random.seed(model["seed"])

    # Generate the activation functions lambdas from their names

    hidden_activation_functions = [activation_function_from_name(name) for (name, _) in conf["hidden_layers"][0]]
    hidden_activation_functions.append(activation_function_from_name(conf["hidden_layers"][1]))

    # Extract the sizes from the (function_name, size) tuples describing the network topology

    hidden_layers_sizes = [size for (_, size) in conf["hidden_layers"][0]]

    # Initialize the network layers with Dense fully-connected layers:

    # Input layer

    model["layers"].append(Dense(in_dimension, hidden_layers_sizes[0], activation_func=hidden_activation_functions[0], weights_init=conf['weights_init']))

    # Hidden layers

    for i in range(len(hidden_layers_sizes) - 1):
        model["layers"].append(Dense(hidden_layers_sizes[i], hidden_layers_sizes[i + 1], activation_func=hidden_activation_functions[i + 1], weights_init=conf['weights_init']))

    # Output layer

    model["layers"].append(Dense(hidden_layers_sizes[-1], out_dimension, activation_func=hidden_activation_functions[-1], weights_init=conf['weights_init']))

    return model

def predict(model, x):
    """
    Main constructor for a sequentially connected whole network.
    The network will be composed of Layers, each with its bias and weight.

    :param conf: configuration dictionary with the network structure and settings
    :return: a dictionary object representing the network and its layers
    """

    # Simply call the forward function of each layer repeatedly,
    # chaining each output as the input of the next one.

    for layer in model["layers"]:
        x = forward(layer, x)
    return x
