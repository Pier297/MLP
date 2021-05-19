import numpy as np
from MLP.Metrics import MSE
from math import ceil

# Early stopping:
# The function works by applying gradient descent onto the training set,
# and each epoch it checks the error on the validation set to see if it
# improved. In case it got better we save the current configuration (epochs, ..),
# otherwise we keep trying for a 'MAX_UNLUCKY_STEPS'. If after
# 'MAX_UNLUCKY_STEPS' the validation error hasn't reached a new minima, we return
# the model trained on the best configuration found on the union of the
# training and validation set.
def early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS = 10):
    # Check input and output dimensions of the model to see that it checks the data.
    input_dimension = model.layers[0].dimension_in
    output_dimension = model.layers[-1].dimension_out
    assert input_dimension == train_X[0].shape[0], f'Input dimension ({input_dimension}) of the model don\'t match with the input dimension ({train_X[0].shape[0]}) of the data.'
    assert output_dimension == train_Y[0].shape[0], f'Output dimension ({output_dimension}) of the model don\'t match with the Output dimension ({train_Y[0].shape[0]}) of the data.'

    best_number_of_epochs = _early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS)

    X = np.row_stack((train_X, val_X))
    Y = np.row_stack((train_Y, val_Y))

    model = model.reset()
    # Train on training and validation data
    return optimizer.optimize(model, X, Y, best_number_of_epochs)

def _early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS):
        best_val_accuracy = MSE(model, val_X, val_Y)
        best_number_of_epochs = 0
        unlucky_steps = 0
        
        optimizer.prev_delta_W = [np.zeros(layer.W.shape) for layer in model.layers]
        optimizer.prev_delta_b = [np.zeros(layer.b.shape) for layer in model.layers]

        for t in range(optimizer.MAX_EPOCHS):
            # Shuffle the training data
            dataset = np.column_stack((train_X, train_Y))
            dataset = np.random.permutation(dataset)
            # For each minibatch apply gradient descent
            for i in range(0, ceil(dataset.shape[0] / optimizer.BATCH_SIZE)):
                # TODO: This sampling is not i.i.d.
                mini_batch = dataset[i * optimizer.BATCH_SIZE:(i * optimizer.BATCH_SIZE) + optimizer.BATCH_SIZE][:]
                optimizer.step(model, mini_batch)
            
            current_val_accuracy = MSE(model, val_X, val_Y)
            if current_val_accuracy < best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_number_of_epochs = t + 1
                unlucky_steps = 0
            elif unlucky_steps == MAX_UNLUCKY_STEPS:
                break # Early stopping
            else:
                unlucky_steps += 1

        return best_number_of_epochs