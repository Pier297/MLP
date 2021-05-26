import numpy as np

class MSE:
    def __init__(self):
        self.name = 'MSE'

    def eval(self, model, X, Y):
        e = 0.0
        for i in range(X.shape[0]):
            e += np.sum((Y[i] - model.predict(X[i]))**2)
        return e / (2 * X.shape[0])

    def loss(self, model, X, Y, lam):
        err = self.eval(model, X, Y)
        squared_weights_sum = 0.0
        for layer in model.layers:
            for row_weights in layer.W:
                squared_weights_sum += np.linalg.norm(row_weights) ** 2
        return err + ((lam / 2) * squared_weights_sum)

    def gradient(self, last_layer_output, target):
        return last_layer_output - target

class CrossEntropy:
    def __init__(self):
        self.name = 'Cross Entropy'

    def eval(self, model, X, Y):
        c = 0.0
        n = X.shape[0] # number of training samples
        for i in range(n):
            a = model.predict(X[i])
            y = Y[i]
            c += np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return -1/n * c

    def gradient(self, last_layer_output, target):
        y = target
        o = last_layer_output
        return (y - 1) / (o - 1) - y/o

def accuracy(model, X, Y) -> float:
    corrects = 0
    for i in range(X.shape[0]):
        if Y[i] == (1 if model.predict(X[i]) >= 0 else -1):
            corrects += 1
    return corrects / X.shape[0]
