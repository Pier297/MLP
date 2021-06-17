from multiprocessing import Value
import numpy as np
from MLP.Network import predict

class MSE:
    def __init__(self):
        self.name = 'MSE'

    def eval(self, model, dataset):
        e = 0.0
        # TODO: Create predict_batch to avoid this for-loop
        for pattern in dataset:
            x = pattern[:model["in_dimension"]]
            y = pattern[model["in_dimension"]:]
            e += np.sum((y - predict(model, x))**2)
        return e / (2 * x.shape[0])

    def loss(self, model, dataset, lam):
        err = self.eval(model, dataset)
        squared_weights_sum = 0.0
        for layer in model["layers"]:
            for row_weights in layer["W"]:
                squared_weights_sum += np.linalg.norm(row_weights) ** 2
        return err + ((lam / 2) * squared_weights_sum)

    def gradient(self, last_layer_output, target):
        return last_layer_output - target


EPSILON = 1e-9

class CrossEntropy:
    def __init__(self):
        self.name = 'Cross Entropy'

    def eval(self, model, dataset, epoch=-1):
        c = 0.0
        for pattern in dataset:
            x = pattern[:model["in_dimension"]]
            a = predict(model, x)
            y = pattern[model["in_dimension"]:]
            c += np.sum(y * np.log(a + EPSILON) + (1 - y) * np.log(1 - a + EPSILON))
        return -1/dataset.shape[0] * c

    def gradient(self, last_layer_output, target):
        y = target
        o = last_layer_output
        return (y - 1) / ((o - 1) + EPSILON) - y/(o + EPSILON)

def loss_function_from_name(name):
    if name == 'Cross Entropy':
        return CrossEntropy()
    elif name == 'MSE':
        return MSE()
    else:
        raise ValueError("Invalid loss function name.")


def accuracy(model, dataset, target_domain=(-1, 1)) -> float:
    corrects = 0
    for i in range(dataset.shape[0]):
        x = dataset[i][:model["in_dimension"]]
        y = dataset[i][model["in_dimension"]:]
        if y == (target_domain[1] if predict(model, x) >= (target_domain[0] + target_domain[1])/2 else target_domain[0]):
            corrects += 1
    return corrects / dataset.shape[0]