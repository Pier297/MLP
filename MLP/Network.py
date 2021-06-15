import numpy as np
from MLP.Layers import Dense, forward
from MLP.ActivationFunctions import activation_function_from_name

def Sequential(conf):
    in_dimension = conf["in_dimension"]
    out_dimension = conf["out_dimension"]

    model = {}
    model["seed"] = conf["seed"]
    model["layers"] = []
    model["in_dimension"] = in_dimension
    model["out_dimension"] = out_dimension

    # Generate the activation functions lambda from their names
    # ([('tanh',4)],'sigmoid')
    hidden_activation_functions = [activation_function_from_name(name) for (name, _) in conf["hidden_layers"][0]]
    hidden_activation_functions.append(activation_function_from_name(conf["hidden_layers"][1]))

    hidden_layers_sizes = [size for (_, size) in conf["hidden_layers"][0]]

    np.random.seed(model["seed"])

    model["layers"].append(Dense(in_dimension, hidden_layers_sizes[0], activation_func=hidden_activation_functions[0]))
    for i in range(len(hidden_layers_sizes) - 1):
        model["layers"].append(Dense(hidden_layers_sizes[i], hidden_layers_sizes[i + 1], activation_func=hidden_activation_functions[i + 1]))
    model["layers"].append(Dense(hidden_layers_sizes[-1], out_dimension, activation_func=hidden_activation_functions[-1]))

    return model


def predict(model, x):
    for layer in model["layers"]:
        x = forward(layer, x)
    return x


def reset(old_model):
    model = {}
    model["layers"] = []
    model["in_dimension"] = old_model["in_dimension"]
    model["out_dimension"] = old_model["out_dimension"]

    for layer in old_model["layers"]:
        model["layers"].append(Dense(layer["in_dimension"], layer["out_dimension"], layer["use_bias"], (layer["activation_func"], layer["activation_func_derivative"])))

    return model