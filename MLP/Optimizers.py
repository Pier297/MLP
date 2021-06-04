from math import ceil
import numpy as np
from MLP.LossFunctions import accuracy, MSE, CrossEntropy
from MLP.Layers import forward, net

def print_epoch_stats(loss_func, model, t, training, validation):
    print(f'Epoch {t+1} | Train {loss_func.name} = {loss_func.eval(model, training)} | Validation {loss_func.name} = {loss_func.eval(model, validation)}')

# Returns (train_errors, val_errors, train_accuracies, val_accuracies)
def Gradient_descent(model, training, validation, loss_function, lr: float, l2: float, momentum: float, batch_percentage: int = 1.0, MAX_EPOCHS: int = 100, target_domain=(-1, 1)):
    train_errors = [loss_function.eval(model, training)]
    train_accuracies = [accuracy(model, training, target_domain=target_domain)]
    val_errors = [loss_function.eval(model, validation)]
    val_accuracies = [accuracy(model, validation, target_domain=target_domain)]

    # Initialize to 0 the prev delta (used for momentum)
    prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
    prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]

    batch_size = int(training.shape[0] * batch_percentage)

    for t in range(MAX_EPOCHS):
        # Shuffle the training data
        # TODO: Sample the mini-batches correctly? This sampling is not i.i.d.
        dataset = np.random.permutation(dataset)
        # For each minibatch apply gradient descent
        for i in range(0, ceil(dataset.shape[0] / batch_size)):
            mini_batch = dataset[i * batch_size:(i * batch_size) + batch_size][:]
            # Perform a gradient descent update on the mini-batch
            gradient_descent_step(model, mini_batch, batch_size, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b)

        train_errors.append(loss_function.eval(model, training))
        val_errors.append(loss_function.eval(model, validation))
        train_accuracies.append(accuracy(model, training, target_domain=target_domain))
        val_accuracies.append(accuracy(model, validation, target_domain=target_domain))
        print_epoch_stats(loss_function, model, t, training, validation)
    return (train_errors, train_accuracies, val_errors, val_accuracies)


def gradient_descent_step(model, mini_batch, batch_size, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b):
    # TODO: batch_size can be computed here, no need to be a parameter.
    nabla_W = []
    nabla_b = []
    for layer in model["layers"]:
        nabla_W.append(np.zeros(layer["W"].shape))
        nabla_b.append(np.zeros(layer["b"].shape))
    input_dimension = model["layers"][0]["dimension_in"]

    # Compute the gradient
    for point in mini_batch:
        x = point[:input_dimension].T
        y = point[input_dimension:].T
        (new_nabla_W, new_nabla_b) = backpropagate(model, x, y, loss_function)
        # TODO: By making nabla_W and nabla_b tensors we can just sum them without having to iterate
        #       then we do this iteration only once when we update the weights outside this loop.
        #       (ofc make nabla_W & nabla_b also tensors in the backpropage method)
        for i in range(len(nabla_W)):
            nabla_W[i] += new_nabla_W[i]
            nabla_b[i] += new_nabla_b[i]

    # Update the model parameters
    for i in range(len(nabla_W)):
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr/BATCH_SIZE * nabla_W[i]
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr/BATCH_SIZE * nabla_b[i]

        model["layers"][i]["W"] += prev_delta_W[i] - l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += prev_delta_b[i] - l2 * model["layers"][i]["b"]


def backpropagate(model, x, y, loss_function):
    # Forward computation
    delta_W = []
    delta_b = []
    activations = [x]
    net_activation_derivative = []
    for layer in model["layers"]:
        delta_W.append(np.zeros(layer["W"].shape))
        delta_b.append(np.zeros(layer["b"].shape))
        net_activation_derivative.append(layer["activation_func_derivative"](net(layer, x)))
        x = forward(layer, x)
        activations.append(x)

    # Backward computation
    # Output layer
    d_error_over_d_output = loss_function.gradient(activations[-1], y)
    d_E_over_d_net = d_error_over_d_output * net_activation_derivative[-1]
    delta_W[-1] += d_E_over_d_net * activations[-1]
    delta_b[-1] += d_E_over_d_net

    # Hidden layer: iterate from the last hidden layer to the first
    for l in range(2, len(model["layers"]) + 1):
        d_error_over_d_output = np.dot(model["layers"][-l + 1]["W"].T, d_E_over_d_net)
        d_E_over_d_net = d_error_over_d_output * net_activation_derivative[-l]
        delta_b[-l] += d_E_over_d_net
        delta_W[-l] += np.dot(d_E_over_d_net, np.reshape(activations[-l - 1], (model["layers"][-l]["dimension_in"], 1)).T)

    return (delta_W, delta_b)