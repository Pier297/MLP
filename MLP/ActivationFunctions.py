import numpy as np

def Tanh():
    return (np.tanh, lambda x: (1 - np.tanh(x)**2))

def Sigmoid():
    sigm = lambda x: 1 / (1 + np.exp(-x))
    der = lambda x: sigm(x) * (1 - sigm(x))
    return (sigm, der)