from MLP.Layers import Dense, forward

def Sequential(configuration, in_dimension, out_dimension):
    model = {}
    model["layers"] = []
    # Add weights
    model["layers"].append(Dense(in_dimension, configuration["hidden_layers"][0]))
    for i in range(len(configuration["hidden_layers"]) - 1):
        model["layers"].append(Dense(configuration["hidden_layers"][i], configuration["hidden_layers"][i + 1]))
    model["layers"].append(Dense(configuration["hidden_layers"][-1], out_dimension))

    return model


def predict(model, x):
    for layer in model["layers"]:
        x = forward(layer, x)
    return x


def reset(old_model):
    model = {}
    model["layers"] = []

    for layer in old_model["layers"]:
        model["layers"].append(Dense(layer["dimension_in"], layer["dimension_out"], layer["use_bias"], (layer["activation_func"], layer["activation_func_derivative"])))
    
    return model