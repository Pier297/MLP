from math import ceil
import numpy as np
from MLP.Metrics import accuracy

def print_epoch_stats(loss_func, model, t, train_X, train_Y, val_X, val_Y):
    train_accuracy = accuracy(model, train_X, train_Y)
    val_accuracy = accuracy(model, val_X, val_Y)
    print(f'Epoch {t+1} | MSE = {loss_func.eval(model, train_X, train_Y)} | Training accuracy = {train_accuracy} | Validation accuracy = {val_accuracy}')


def print_epoch_stats(loss_func, model, t, X, Y):
    train_accuracy = accuracy(model, X, Y)
    print(f'Epoch {t+1} | MSE = {loss_func.eval(model, X, Y)} | Training accuracy = {train_accuracy}')


# Gradient Descent optimization supporting
# Online or Batch learning and Early Stopping Regularization
class GradientDescent:
    # TODO: Maybe change MLP.LossFunctions.MSE to a function instead of a class, remove the L2 parameter from it
    #       and pass it here. then we can call 'loss_function(model, L2, X, Y)' for example to compute the MSE
    def __init__(self, loss_function, lr: float = 0.01, momentum: float = 0.5, BATCH_SIZE = 10):
        self.loss_function = loss_function
        self.lr = lr
        self.momentum = momentum
        self.BATCH_SIZE = BATCH_SIZE

    def optimize(self, model, X, Y, MAX_EPOCHS):
        train_errors = [self.loss_function.eval(model, X, Y)]
        train_accuracies = [accuracy(model, X, Y)]

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
            print_epoch_stats(self.loss_function, model, t, X, Y)
        return (train_errors, train_accuracies)

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
                delta_W[i] += new_delta_W[i] + (self.loss_function.L2 * model.layers[i].W)
                delta_b[i] += new_delta_b[i] + (self.loss_function.L2 * model.layers[i].b)
        
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
            net_activation_derivative.append(layer.activation_func_derivative(layer.net(x)))
            x = layer.forward(x)
            activations.append(x)

        # Backward computation
        # Output layer
        d_error_over_d_output = activations[-1] - y
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