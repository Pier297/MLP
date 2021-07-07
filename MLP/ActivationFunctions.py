import numpy as np

# ActivationFunctions.py
# Definitions of the main activation functions available in the network.
# These classes contain both the actual function and its derivative.
# Self-explanatory. 

def Tanh():
    return (np.tanh, lambda x: (1 - np.tanh(x)**2))

def Sigmoid():
    sigm = lambda x: 1 / (1 + np.exp(-x))
    der = lambda x: sigm(x) * (1 - sigm(x))
    return (sigm, der)

def Linear():
    return (lambda x: x, lambda _: 1)

def Relu():
    return (lambda x: np.maximum(x, 0.0), lambda x: 1.0 * (x > 0.0))

def LeakyRelu():
    return (lambda x: x * (x > 0.0) + 0.01 * x * (x <= 0.0), lambda x : (x > 0) * 1 + (x <= 0) * 0.01)

def Softplus():
    return (lambda x: np.log(1 + np.exp(x)), lambda x: np.exp(x) / (1 + np.exp(x)))

def activation_function_from_name(name):
    """
    Get the activation function from its simple string name.
    Simply consult this straightforward function to read the names.
    :param name: simple string name of the function
    :return: activation function class
    """
    if name == 'tanh':
        return Tanh()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'linear':
        return Linear()
    elif name == 'relu':
        return Relu()
    elif name == 'leaky-relu':
        return LeakyRelu()
    elif name == 'softplus':
        return Softplus()
    else:
        raise ValueError("Invalid activation function name.")