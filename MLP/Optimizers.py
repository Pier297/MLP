from math import ceil
import numpy as np

# TODO: Create base class Optimizer from which we derive the shared method 'optimize'

# Gradient Descent optimization supporting
# Online or Batch learning and Early Stopping Regularization
class GradientDescent:
    def __init__(self, lr: float = 0.01, MAX_EPOCHS = 100, BACTH_SIZE = 10, MAX_UNLUCKY_STEPS = 1):
        self.lr = lr
        self.MAX_EPOCHS = MAX_EPOCHS
        self.BATCH_SIZE = BACTH_SIZE
        self.MAX_UNLUCKY_STEPS = MAX_UNLUCKY_STEPS
    
    # The function works by applying gradient descent onto the training set,
    # and each epoch it checks the error on the validation set to see if it
    # improved. In case it got better we save the current configuration,
    # otherwise we keep trying for a 'MAX_UNLUCKY_STEPS'. If after
    # 'MAX_UNLUCKY_STEPS' the validation error hasn't reached a new low, we return
    # the model trained on the best configuration found on the union of the
    # training and validation set.
    def optimize(self, model, train_X, train_Y, val_X, val_Y):
        # Check that the input and output dimensions of the model matches the one of the data supplied
        input_dimension = model.layers[0].dimension_in
        output_dimension = model.layers[-1].dimension_out
        assert input_dimension == train_X[0].shape[0], f'Input dimension ({input_dimension}) of the model don\'t match with the input dimension ({train_X[0].shape[0]}) of the data.'
        assert output_dimension == train_Y[0].shape[0], f'Input dimension ({output_dimension}) of the model don\'t match with the input dimension ({train_Y[0].shape[0]}) of the data.'

        for t in range(self.MAX_EPOCHS):
            # Shuffle the training data
            dataset = np.column_stack((train_X, train_Y))
            dataset = np.random.permutation(dataset)
            # For each minibatch apply gradient descent
            for i in range(0, ceil(dataset.shape[0] / self.BATCH_SIZE)):
                # TODO: This sampling is not i.i.d.
                mini_batch = dataset[i * self.BATCH_SIZE:(i * self.BATCH_SIZE) + self.BATCH_SIZE][:]
                self.step(model, mini_batch)
            # TODO: Print error and check validation unlucky..
            print(f'Epoch {t+1} | MSE = {round(self.MSE(model, train_X, train_Y)[0][0], 5)} | Training accuracy = {round(self.accuracy(model, train_X, train_Y), 3)} | Validation accuracy = {round(self.accuracy(model, val_X, val_Y), 3)}')

    def accuracy(self, model, X, Y):
        corrects = 0
        for i in range(X.shape[0]):
            if Y[i] == (1 if model.predict(X[i]) >= 0 else -1):
                corrects += 1
        return corrects / X.shape[0]

    def MSE(self, model, X, Y):
        e = 0.0
        for i in range(X.shape[0]):
            e += (Y[i] - model.predict(X[i]))**2
        return e / (2 * X.shape[0])

    def step(self, model, mini_batch):
        delta_W = []
        delta_b = []
        for layer in model.layers:
            delta_W.append(np.zeros(layer.W.shape))
            delta_b.append(np.zeros(layer.b.shape))
        input_dimension = model.layers[0].dimension_in

        for point in mini_batch:
            x = point[:input_dimension].T
            y = point[input_dimension:].T
            (new_delta_W, new_delta_b) = self.backpropagate(model, x, y)
            # TODO: By making delta_W and delta_b tensors we can just sum them without having to iterate
            #       then we do this iteration only once when we update the weights outside this loop.
            #       (ofc make delta_W & delta_b also tensors in the backpropage method)
            for i in range(len(delta_W)):
                delta_W[i] = delta_W[i] + new_delta_W[i]
                delta_b[i] = delta_b[i] + new_delta_b[i]
        
        for i, layer in enumerate(model.layers):
            layer.W = layer.W - (1/(self.lr * self.BATCH_SIZE) * delta_W[i])
            if layer.use_bias:
                layer.b = layer.b - (1/(self.lr * self.BATCH_SIZE) * delta_b[i])

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