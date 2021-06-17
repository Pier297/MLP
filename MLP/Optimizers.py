from math import ceil
import numpy as np
from MLP.LossFunctions import loss_function_from_name, accuracy, MSE, CrossEntropy
from MLP.Layers import forward, net
from math import inf

def print_epoch_stats(loss_function, model, t, training, validation):
    if validation.shape[0] > 0:
        msg = f' | Validation {loss_function.name} = {loss_function.eval(model, validation)}'
    else:
        msg = ''
    print(f'Epoch {t+1} | Train {loss_function.name} = {loss_function.eval(model, training)}' + msg)

# Early stopping:
# The function works by applying gradient descent onto the training set,
# and each epoch it checks the error on the validation set to see if it
# improved. In case it got better we save the current configuration (epochs, ..),
# otherwise we keep trying for a 'max_unlucky_epochs'. If after
# 'max_unlucky_epochs' the validation error hasn't reached a new minima, we return
# the model trained on the best configuration found on the union of the
# training and validation set.
# Returns (train_errors, val_errors, train_accuracies, val_accuracies)
def gradient_descent(model, training, validation, config):

    target_domain         = config['target_domain']
    loss_function         = loss_function_from_name(config['loss_function_name'])
    lr                    = config['lr']
    l2                    = config['l2']
    momentum              = config['momentum']
    mini_batch_percentage = config['mini_batch_percentage']
    max_unlucky_epochs    = config['max_unlucky_epochs']
    max_epochs            = config['max_epochs']
    print_stats           = config['print_stats']

    train_errors = [loss_function.eval(model, training)]
    train_accuracies = [accuracy(model, training, target_domain=target_domain)]
    if validation.shape[0] > 0:
        val_errors = [loss_function.eval(model, validation)]
        val_accuracies = [accuracy(model, validation, target_domain=target_domain)]
    else:
        val_errors = []
        val_accuracies = []

    # Initialize to 0 the prev delta (used for momentum)
    prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
    prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]

    unlucky_epochs = 0
    best_epoch = 0
    best_val_error = inf

    batch_size = int(mini_batch_percentage * training.shape[0])

    for t in range(max_epochs):
        # Shuffle the training data
        # TODO: Sample the mini-batches correctly? This sampling is not i.i.d.
        dataset = np.random.permutation(training)
        # For each minibatch apply gradient descent
        for i in range(0, ceil(dataset.shape[0] / batch_size)):
            mini_batch = dataset[i * batch_size:(i * batch_size) + batch_size][:]
            # Perform a gradient descent update on the mini-batch
            gradient_descent_step(model, mini_batch, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b)

        # Early stopping logic
        if validation.shape[0] > 0:
            current_val_error = loss_function.eval(model, validation, t)

            if current_val_error < best_val_error:
                best_val_error = current_val_error
                best_epoch = t
                unlucky_epochs = 0
            elif unlucky_epochs == max_unlucky_epochs:
                break
            else:
                unlucky_epochs += 1

        train_errors.append(loss_function.eval(model, training, t))
        train_accuracies.append(accuracy(model, training, target_domain=target_domain))
        if validation.shape[0] > 0:
            val_errors.append(loss_function.eval(model, validation, t))
            val_accuracies.append(accuracy(model, validation, target_domain=target_domain))
        
        if print_stats:
            print_epoch_stats(loss_function, model, t, training, validation)
        
    return {'val_error':        val_errors[best_epoch] if validation.shape[0] > 0 else None,
            'best_epoch':       best_epoch,
            'train_errors':     train_errors,
            'val_errors':       val_errors,
            'train_accuracies': train_accuracies,
            'val_accuracies':   val_accuracies}


def gradient_descent_step(model, mini_batch, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b):
    batch_size = mini_batch.shape[0]
    nabla_W = []
    nabla_b = []
    for layer in model["layers"]:
        nabla_W.append(np.zeros(layer["W"].shape))
        nabla_b.append(np.zeros(layer["b"].shape))
    input_dimension = model["in_dimension"]

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
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr/batch_size * nabla_W[i]
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr/batch_size * nabla_b[i]

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
        delta_W[-l] += np.dot(d_E_over_d_net, np.reshape(activations[-l - 1], (model["layers"][-l]["in_dimension"], 1)).T)

    return (delta_W, delta_b)