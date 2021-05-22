import numpy as np

class MSE:
    def __init__(self, L2):
        self.L2 = L2

    def eval(self, model, X, Y):
        e = 0.0
        for i in range(X.shape[0]):
            e += (Y[i] - model.predict(X[i]))**2
        squared_weights_sum = 0.0
        for layer in model.layers:
            for row_weights in layer.W:
                squared_weights_sum += np.linalg.norm(row_weights) ** 2
        return (e / (2 * X.shape[0]))[0][0] + ((self.L2 / 2) * squared_weights_sum)