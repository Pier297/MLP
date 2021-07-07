import numpy as np

# LossFunctions.py
# Contains the main loss functions used in the project, both
# for model evaluation and for learning.

def loss_function_from_name(name):
    """
    Get the loss function function from its simple string name.
    Simply consult this straightforward function to read the names.
    :param name: simple string name of the function
    :return: loss function class
    """
    if name == 'Cross Entropy':
        return CrossEntropy()
    elif name == 'MSE':
        return MSE()
    else:
        raise ValueError("Invalid loss function name.")

class MSE:
    """
    Mean Squared Error loss function. Self-explanatory.
    """
    def __init__(self):
        self.name = 'MSE'

    def eval(self, output, target):
        return np.average((target - output)**2)

    def std(self, output, target):
        return np.std((target - output)**2)

    def gradient(self, last_layer_output, target):
        return 2 * (last_layer_output - target)

CROSS_ENTROPY_EPSILON = 1e-9

class CrossEntropy:
    """
    Mean Squared Error loss function. Self-explanatory.
    """
    def __init__(self):
        self.name = 'Cross Entropy'

    def eval(self, output, target):
        # Add CROSS_ENTROPY_EPSILON to the numerator for increased numerical stability.
        return -np.average(target * np.log(output + CROSS_ENTROPY_EPSILON) + (1 - target) * np.log(1 - output + CROSS_ENTROPY_EPSILON))

    def std(self, output, target):
        # Add CROSS_ENTROPY_EPSILON to the numerator for increased numerical stability.
        return np.std(target * np.log(output + CROSS_ENTROPY_EPSILON) + (1 - target) * np.log(1 - output + CROSS_ENTROPY_EPSILON))

    def gradient(self, last_layer_output, target):
        # Add CROSS_ENTROPY_EPSILON to the numerator for increased numerical stability.
        return (target - 1) / ((last_layer_output - 1) + CROSS_ENTROPY_EPSILON) - target/(last_layer_output + CROSS_ENTROPY_EPSILON)

def discretize(output, target_domain=(-1, 1)):
    """
    Helper function for the discretization of a given output vector
    according to the target domain specified.
    :param output: the output given by the model
    :param target_domain: the reference output domain used for the problem
    :return: discretized output within the target_domain
    """
    return np.array([target_domain[1] if v >= (target_domain[0] + target_domain[1])/2 else target_domain[0] for v in output])

def accuracy(output, target, target_domain) -> float:
    """
    Accuracy percentage function. The output is first discretized in order to be
    compared with the target vector.
    :param output: the output given by the model
    :param target: the target values of the dataset
    :param target_domain: the reference output domain used for the problem
    :return: accuracy percentage of the output with respect to the discrete target
    """
    corrects = 0
    discrete_output = discretize(output, target_domain)
    for i in range(output.shape[0]):
        # If the output is correct, add one to the correct values.
        if target[i] == discrete_output[i]:
            corrects += 1
    return corrects / output.shape[0]

def mean_euclidean_error(output, target):
    """
    Mean Euclidean Error function given just as performance evaluation.
    No training class is provided for this function and is only available as direct evaluation.
    :param output: the output given by the model
    :param target: the target values of the dataset
    :return: Mean Euclidean Error computed
    """
    return np.average(np.sqrt((target[:, 0] - output[:, 0])**2 + (target[:, 1] - output[:, 1])**2))

def mean_euclidean_error_var(output, target):
    """
    Mean Euclidean Error function given just as performance evaluation.
    No training class is provided for this function and is only available as direct evaluation.
    :param output: the output given by the model
    :param target: the target values of the dataset
    :return: Variance of the Euclidean Error
    """
    return np.var(np.sqrt((target[:, 0] - output[:, 0])**2 + (target[:, 1] - output[:, 1])**2))
