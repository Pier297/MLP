import numpy as np

class MSE:
    def __init__(self, L2):
        self.L2 = L2
        self.name = 'MSE'

    def eval(self, model, X, Y):
        e = 0.0
        for i in range(X.shape[0]):
            e += np.sum((Y[i] - model.predict(X[i]))**2)
        squared_weights_sum = 0.0
        for layer in model.layers:
            for row_weights in layer.W:
                squared_weights_sum += np.linalg.norm(row_weights) ** 2
        return (e / (2 * X.shape[0])) + ((self.L2 / 2) * squared_weights_sum)

    def gradient(self, last_layer_output, target):
        return last_layer_output - target


class CrossEntropy:
    def __init__(self, L2=0):
        self.L2 = L2 # Here because we call a generic 'loss_func.L2'
        self.name = 'Cross Entropy'

    def eval(self, model, X, Y):
        C = 0.0
        n = X.shape[0] # number of training samples
        for i in range(n):
            a = model.predict(X[i])
            y = Y[i]
            C += np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return -1/n * C

    def gradient(self, last_layer_output, target):
        return -((target / last_layer_output) - ((1-target)/(1-last_layer_output)))