from math import ceil
import numpy as np
from MLP.LossFunctions import accuracy, MSE, CrossEntropy
from MLP.Layers import forward, net

def print_epoch_stats(loss_func, model, t, train_X, train_Y, val_X, val_Y):
    print(f'Epoch {t+1} | Train {loss_func.name} = {loss_func.eval(model, train_X, train_Y)} | Validation {loss_func.name} = {loss_func.eval(model, val_X, val_Y)}')


""" def print_epoch_stats(loss_func, model, t, X, Y):
    train_accuracy = accuracy(model, X, Y)
    print(f'Epoch {t+1} | {loss_func.name} = {loss_func.eval(model, X, Y)} | Training accuracy = {train_accuracy}') """


# Returns (train_errors, val_errors)
def Gradient_descent(model, train_X, train_Y, val_X, val_Y, loss_function, lr: float, l2: float, momentum: float, BATCH_SIZE: int = 10, MAX_EPOCHS: int = 100):
    train_errors = [loss_function.eval(model, train_X, train_Y)]
    val_errors = [loss_function.eval(model, val_X, val_Y)]

    # Initialize to 0 the prev delta (used for momentum)
    prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
    prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]

    for t in range(MAX_EPOCHS):
        # Shuffle the training data
        # TODO: Sample the mini-batches correctly? This sampling is not i.i.d.
        dataset = np.column_stack((train_X, train_Y))
        dataset = np.random.permutation(dataset)
        # For each minibatch apply gradient descent
        for i in range(0, ceil(dataset.shape[0] / BATCH_SIZE)):
            mini_batch = dataset[i * BATCH_SIZE:(i * BATCH_SIZE) + BATCH_SIZE][:]
            # Perform a gradient descent update on the mini-batch
            step(model, mini_batch, BATCH_SIZE, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b)

        train_errors.append(loss_function.eval(model, train_X, train_Y))
        val_errors.append(loss_function.eval(model, val_X, val_Y))
        print_epoch_stats(loss_function, model, t, train_X, train_Y, val_X, val_Y)
    return (train_errors, val_errors)


def step(model, mini_batch, BATCH_SIZE, loss_function, lr, l2, momentum, prev_delta_W, prev_delta_b):
    # TODO: BATCH_SIZE can be computed here, no need to be a parameter.
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
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr * (nabla_W[i]/BATCH_SIZE + (l2 * model["layers"][i]["W"]))
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr * (nabla_b[i]/BATCH_SIZE + (l2 * model["layers"][i]["b"]))

        model["layers"][i]["W"] += prev_delta_W[i]
        model["layers"][i]["b"] += prev_delta_b[i]


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


# Gradient Descent optimization supporting
# Online or Batch learning
class GradientDescent:
    def __init__(self, loss_function, lr: float = 0.01, l2: float = 0.0, momentum: float = 0.5, BATCH_SIZE = 10):
        self.loss_function = loss_function
        self.lr = lr
        self.l2 = l2
        self.momentum = momentum
        self.BATCH_SIZE = BATCH_SIZE

    def optimize(self, model, X, Y, val_X, val_Y, MAX_EPOCHS):
        train_errors = [self.loss_function.eval(model, X, Y)]
        train_accuracies = [accuracy(model, X, Y)]
        val_errors = [self.loss_function.eval(model, val_X, val_Y)]
        val_accuracies = [accuracy(model, val_X, val_Y)]

        # Initialize to 0 the prev delta (used for nesterov momentum)
        self.prev_delta_W = [np.zeros(layer.W.shape) for layer in model.layers]
        self.prev_delta_b = [np.zeros(layer.b.shape) for layer in model.layers]

        for t in range(MAX_EPOCHS):
            # Shuffle the training data
            # TODO: Sample the mini-batches
            dataset = np.column_stack((X, Y))
            dataset = np.random.permutation(dataset)
            # For each minibatch apply gradient descent
            for i in range(0, ceil(dataset.shape[0] / self.BATCH_SIZE)):
                # TODO: This sampling is not i.i.d.
                mini_batch = dataset[i * self.BATCH_SIZE:(i * self.BATCH_SIZE) + self.BATCH_SIZE][:]
                self.step(model, mini_batch)

            train_errors.append(self.loss_function.eval(model, X, Y))
            train_accuracies.append(accuracy(model, X, Y))
            val_errors.append(self.loss_function.eval(model, val_X, val_Y))
            val_accuracies.append(accuracy(model, val_X, val_Y))
            print_epoch_stats(self.loss_function, model, t, X, Y)
        return (train_errors, train_accuracies, val_errors, val_accuracies)

    
    def step(self, model, mini_batch):
        delta_W = []
        delta_b = []
        for layer in model.layers:
            delta_W.append(np.zeros(layer.W.shape))
            delta_b.append(np.zeros(layer.b.shape))
        input_dimension = model.layers[0].dimension_in

        # theta
        old_layers_W = []
        old_layers_b = []
        for i, layer in enumerate(model.layers):
            old_layers_W.append(layer.W)
            old_layers_b.append(layer.b)

        # Apply interim update:
        for i, layer in enumerate(model.layers):
            layer.W = layer.W + self.momentum * self.prev_delta_W[i]
            if layer.use_bias:
                layer.b = layer.b + self.momentum * self.prev_delta_b[i]

        # Compute the gradient
        for point in mini_batch:
            x = point[:input_dimension].T
            y = point[input_dimension:].T
            (new_delta_W, new_delta_b) = self.backpropagate(model, x, y)
            # TODO: By making delta_W and delta_b tensors we can just sum them without having to iterate
            #       then we do this iteration only once when we update the weights outside this loop.
            #       (ofc make delta_W & delta_b also tensors in the backpropage method)
            for i in range(len(delta_W)):
                delta_W[i] += new_delta_W[i] + (self.l2 * model.layers[i].W)
                delta_b[i] += new_delta_b[i] + (self.l2 * model.layers[i].b)

        # Compute velocity update and apply update:
        for i, layer in enumerate(model.layers):
            self.prev_delta_W[i] = self.momentum * self.prev_delta_W[i] - ((self.lr/self.BATCH_SIZE) * delta_W[i])
            layer.W = old_layers_W[i] + self.prev_delta_W[i]
            if layer.use_bias:
                self.prev_delta_b[i] = self.momentum * self.prev_delta_b[i] - ((self.lr/self.BATCH_SIZE) * delta_b[i])
                layer.b = old_layers_b[i] + self.prev_delta_b[i]


    def backpropagate(self, model, x, y):
        # Forward computation
        delta_W = []
        delta_b = []
        activations = [x]
        net_activation_derivative = []
        for layer in model.layers:
            delta_W.append(np.zeros(layer.W.shape))
            delta_b.append(np.zeros(layer.b.shape))
            net_activation_derivative.append(layer.activation_func_derivative(net(layer, x)))
            x = forward(layer, x)
            activations.append(x)

        # Backward computation
        # Output layer
        d_error_over_d_output = self.loss_function.gradient(activations[-1], y)
        d_E_over_d_net = d_error_over_d_output * net_activation_derivative[-1]
        delta_W[-1] += d_E_over_d_net * activations[-1]
        delta_b[-1] += d_E_over_d_net

        # Hidden layer: iterate from the last hidden layer to the first
        for l in range(2, len(model.layers) + 1):
            d_error_over_d_output = np.dot(model.layers[-l + 1].W.T, d_E_over_d_net)
            d_E_over_d_net = d_error_over_d_output * net_activation_derivative[-l]
            delta_b[-l] += d_E_over_d_net
            delta_W[-l] += np.dot(d_E_over_d_net, np.reshape(activations[-l - 1], (model.layers[-l].dimension_in, 1)).T)

        return (delta_W, delta_b)