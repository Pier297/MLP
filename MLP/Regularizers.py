from MLP.Optimizers import step
import numpy as np
from MLP.LossFunctions import accuracy
from math import ceil, inf

# Early stopping:
# The function works by applying gradient descent onto the training set,
# and each epoch it checks the error on the validation set to see if it
# improved. In case it got better we save the current configuration (epochs, ..),
# otherwise we keep trying for a 'MAX_UNLUCKY_STEPS'. If after
# 'MAX_UNLUCKY_STEPS' the validation error hasn't reached a new minima, we return
# the model trained on the best configuration found on the union of the
# training and validation set.
def early_stopping(model, loss_function, lr: float, l2: float, momentum: float, BATCH_SIZE, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS = 10, MAX_EPOCHS = 250, target_domain=(-1, 1)):
    train_errors = [loss_function.eval(model, train_X, train_Y)]
    train_accuracies = [accuracy(model, train_X, train_Y, target_domain=target_domain)]
    val_errors = [loss_function.eval(model, val_X, val_Y)]
    val_accuracies = [accuracy(model, val_X, val_Y, target_domain=target_domain)]
    unlucky_epochs = 0
    best_epoch = 0
    best_val_error = inf

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

        # Early stopping logic
        current_val_error = loss_function.eval(model, val_X, val_Y)
        if current_val_error < best_val_error:
            best_val_error = current_val_error
            best_epoch = t + 1
            unlucky_epochs = 0
        elif unlucky_epochs == MAX_UNLUCKY_STEPS:
            break
        else:
            unlucky_epochs += 1

        train_errors.append(loss_function.eval(model, train_X, train_Y))
        train_accuracies.append(accuracy(model, train_X, train_Y, target_domain=target_domain))
        val_errors.append(current_val_error)
        val_accuracies.append(accuracy(model, val_X, val_Y, target_domain=target_domain))
    return (train_errors, train_accuracies, val_errors, val_accuracies, best_epoch)