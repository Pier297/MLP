import numpy as np

def loss_function_from_name(name):
    if name == 'Cross Entropy':
        return CrossEntropy()
    elif name == 'MSE':
        return MSE()
    else:
        raise ValueError("Invalid loss function name.")

class MSE:
    def __init__(self):
        self.name = 'MSE'

    def eval(self, output, target):
        return np.average((target - output)**2)

    def std(self, output, target):
        return np.std((target - output)**2)

    def gradient(self, last_layer_output, target):
        return 2 * (last_layer_output - target)

EPSILON = 1e-9

class CrossEntropy:
    def __init__(self):
        self.name = 'Cross Entropy'

    def eval(self, output, target):
        return -np.average(target * np.log(output + EPSILON) + (1 - target) * np.log(1 - output + EPSILON))

    def std(self, output, target):
        return np.std(target * np.log(output + EPSILON) + (1 - target) * np.log(1 - output + EPSILON))

    def gradient(self, last_layer_output, target):
        return (target - 1) / ((last_layer_output - 1) + EPSILON) - target/(last_layer_output + EPSILON)

def discretize(output, target_domain=(-1, 1)):
    return np.array([target_domain[1] if v >= (target_domain[0] + target_domain[1])/2 else target_domain[0] for v in output])

def accuracy(output, target, target_domain) -> float:
    corrects = 0
    discrete_output = discretize(output, target_domain)
    for i in range(output.shape[0]):
        if target[i] == discrete_output[i]:
            corrects += 1
    return corrects / output.shape[0]

def mean_euclidean_error(output, target):
    y1_output = output[:, 0]
    y2_output = output[:, 1]
    y1_target = target[:, 0]
    y2_target = target[:, 1]
    return np.average(np.sqrt((y1_target - y1_output)**2 + (y2_target - y2_output)**2))

def mean_euclidean_error_var(output, target):
    y1_output = output[:, 0]
    y2_output = output[:, 1]
    y1_target = target[:, 0]
    y2_target = target[:, 1]
    return np.var(np.sqrt((y1_target - y1_output)**2 + (y2_target - y2_output)**2))
