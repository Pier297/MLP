import numpy as np
from MLP.LossFunctions import accuracy
from math import ceil

# Early stopping:
# The function works by applying gradient descent onto the training set,
# and each epoch it checks the error on the validation set to see if it
# improved. In case it got better we save the current configuration (epochs, ..),
# otherwise we keep trying for a 'MAX_UNLUCKY_STEPS'. If after
# 'MAX_UNLUCKY_STEPS' the validation error hasn't reached a new minima, we return
# the model trained on the best configuration found on the union of the
# training and validation set.
""" #
# Returns a tuple (train_errors, val_errors) containing the history of errors on training & val sets.
def early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 100):
    # Check input and output dimensions of the model to see that it checks the data.
    input_dimension = model.layers[0].dimension_in
    output_dimension = model.layers[-1].dimension_out
    assert input_dimension == train_X[0].shape[0], f'Input dimension ({input_dimension}) of the model don\'t match with the input dimension ({train_X[0].shape[0]}) of the data.'
    assert output_dimension == train_Y[0].shape[0], f'Output dimension ({output_dimension}) of the model don\'t match with the Output dimension ({train_Y[0].shape[0]}) of the data.'

    best_number_of_epochs = early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS, MAX_EPOCHS)

    X = np.row_stack((train_X, val_X))
    Y = np.row_stack((train_Y, val_Y))

    model = model.reset()
    # Train on training and validation data
    return optimizer.optimize(model, X, Y, best_number_of_epochs) """

def early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS, MAX_EPOCHS):
        best_val_error = optimizer.loss_function.eval(model, val_X, val_Y)
        best_number_of_epochs = 0
        unlucky_steps = 0

        train_errors = [optimizer.loss_function.eval(model, train_X, train_Y)]
        train_accuracies = [accuracy(model, train_X, train_Y)]
        val_errors = [optimizer.loss_function.eval(model, val_X, val_Y)]
        val_accuracies = [accuracy(model, val_X, val_Y)]

        optimizer.prev_delta_W = [np.zeros(layer.W.shape) for layer in model.layers]
        optimizer.prev_delta_b = [np.zeros(layer.b.shape) for layer in model.layers]

        for t in range(MAX_EPOCHS):
            # Shuffle the training data
            dataset = np.column_stack((train_X, train_Y))
            dataset = np.random.permutation(dataset)
            # For each minibatch apply gradient descent
            for i in range(0, ceil(dataset.shape[0] / optimizer.BATCH_SIZE)):
                # TODO: This sampling is not i.i.d.
                mini_batch = dataset[i * optimizer.BATCH_SIZE:(i * optimizer.BATCH_SIZE) + optimizer.BATCH_SIZE][:]
                optimizer.step(model, mini_batch)

            train_errors.append(optimizer.loss_function.eval(model, train_X, train_Y))
            train_accuracies.append(accuracy(model, train_X, train_Y))
            val_errors.append(optimizer.loss_function.eval(model, val_X, val_Y))
            val_accuracies.append(accuracy(model, val_X, val_Y))

            current_val_error = optimizer.loss_function.eval(model, val_X, val_Y)
            if current_val_error < best_val_error:
                best_val_error = current_val_error
                best_number_of_epochs = t + 1
                unlucky_steps = 0
            elif unlucky_steps == MAX_UNLUCKY_STEPS:
                break # Early stopping
            else:
                unlucky_steps += 1

        return (best_number_of_epochs, train_errors, train_accuracies, val_errors, val_accuracies)