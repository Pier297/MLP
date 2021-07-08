import numpy as np

# Adam.py

# Implementation of the Adam learning algorithm.

def adam_step(model, epoch, nabla_W, nabla_b, prev_first_delta_W, prev_first_delta_b, prev_second_delta_W, prev_second_delta_b, lr, l2, decay_rate_1, decay_rate_2, delta):
    """
    Weights update procedure for the Adam learning algorithm.
    This is simply called as a replacement for the standard gradient descent weights update.
    Aside from the Adam specific hyperparameters, this function presents the standardized interface for weight updating.
    :param model: model configuration object
    :param epoch: current training epoch
    :param lr: Learning rate
    :param l2: L2 regularization coefficient
    :param nabla_W: gradient computed on the dataset (weights)
    :param nabla_b: gradient computed on the dataset (biases)
    :param prev_first_delta_W:  Adam-specific previous first  moment deltas (weights)
    :param prev_first_delta_b:  Adam-specific previous first  moment deltas (biases)
    :param prev_second_delta_W: Adam-specific previous second moment deltas (weights)
    :param prev_second_delta_b: Adam-specific previous second moment deltas (biases)
    :param decay_rate_1: Adam-specific first moment exponential decay rate
    :param decay_rate_2: Adam-specific second moment exponential decay rate
    :param delta: Coefficient used for numerical stability (recommended ~1e-8)
    """

    # Compute the first and second moment estimations

    prev_first_delta_W  = [decay_rate_1 * prev_first_delta_W[i]  + (1 - decay_rate_1) * nabla_W[i]                for i in range(len(prev_first_delta_W))]
    prev_first_delta_b  = [decay_rate_1 * prev_first_delta_b[i]  + (1 - decay_rate_1) * nabla_b[i]                for i in range(len(prev_first_delta_b))]
    prev_second_delta_W = [decay_rate_2 * prev_second_delta_W[i] + (1 - decay_rate_2) * (nabla_W[i] * nabla_W[i]) for i in range(len(prev_second_delta_W))]
    prev_second_delta_b = [decay_rate_2 * prev_second_delta_b[i] + (1 - decay_rate_2) * (nabla_b[i] * nabla_b[i]) for i in range(len(prev_second_delta_b))]

    correct_first_delta_W  = [s / (1 - decay_rate_1**epoch) for s in prev_first_delta_W]
    correct_first_delta_b  = [s / (1 - decay_rate_1**epoch) for s in prev_first_delta_b]
    correct_second_delta_W = [r / (1 - decay_rate_2**epoch) for r in prev_second_delta_W]
    correct_second_delta_b = [r / (1 - decay_rate_2**epoch) for r in prev_second_delta_b]

    # Update weight and bias parameters for each layer

    for i in range(len(nabla_W)):
        delta_W = -lr * correct_first_delta_W[i] / (np.sqrt(correct_second_delta_W[i]) + delta)
        delta_b = -lr * correct_first_delta_b[i] / (np.sqrt(correct_second_delta_b[i]) + delta)

        model["layers"][i]["W"] += delta_W - l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += delta_b - l2 * model["layers"][i]["b"]
