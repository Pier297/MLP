from MLP.Layers import Dense, forward
from MLP.ActivationFunctions import activation_function_from_name

def Sequential(conf):
    in_dimension = conf["in_dimension"]
    out_dimension = conf["out_dimension"]

    model = {}
    model["layers"] = []
    model["in_dimension"] = in_dimension
    model["out_dimension"] = out_dimension

    # Generate the activation functions lambda from their names
    hidden_activation_functions = [activation_function_from_name(name) for name in conf["hidden_layers_activations"]]

    model["layers"].append(Dense(in_dimension, conf["hidden_layers"][0], activation_func=hidden_activation_functions[0]))
    for i in range(len(conf["hidden_layers"]) - 1):
        model["layers"].append(Dense(conf["hidden_layers"][i], conf["hidden_layers"][i + 1], activation_func=hidden_activation_functions[i + 1]))
    model["layers"].append(Dense(conf["hidden_layers"][-1], out_dimension, activation_func=hidden_activation_functions[-1]))

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