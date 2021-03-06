from math import ceil
import numpy as np
from MLP.LossFunctions import loss_function_from_name, accuracy, mean_euclidean_error
from MLP.Gradient import compute_gradient
from MLP.Network import predict
from MLP.Adam import adam_step
from MLP.Nesterov import SGDN_step
from math import inf
from MLP.Utils import denormalize

# GradientDescent.py

# Implements the main per-epoch training logic with patience and model
# evaluation on validation and a watch dataset, while delegating
# the specific weights update and learning tasks to different optimizers.

def gradient_descent(model, training, validation=None, config={}, watching=None):
    """
    Core function of the network training algorithm.
    Simply extract the training values and hyperparameters from the configuration,
    and then apply the optimizing and weights update strategy (e.g.: Adam, SGD w/ Classical momentum, SGD w/ Nesterov momentum)
    to the network, so that the bulk of the work is delegated and execute in a different module.
    :param model: model data object dictionary
    :param training: training dataset
    :param validation: optional validation dataset on which to perform early stopping
    :param config: hyperparameter configuration
    :param watching: optional additional dataset on which to record performance metrics (e.g.: a test set)
    :return: standard gradient descent results, with the following self-explanatory structure:
       {'best_val_error': float
        'best_epoch': int
        # nparrays of float values:
        'train_errors',  'train_accuracies'
        'val_errors',    'val_accuracies'
        'watch_errors',  'watch_accuracies'
        'weights_norms', 'gradient_norms'
        (used for additional optional metrics)
        'metric_train_errors', 'metric_val_errors', 'metric_watch_errors'
       }
    """

    # Extract the relevant hyperparamters from the hyperconfiguration.

    target_domain              = config['target_domain'] if config['target_domain'] is not None else None
    loss_function              = loss_function_from_name(config['loss_function_name'])
    lr                         = config['lr']
    (lr_final, lr_final_epoch) = config['lr_decay'] if config['lr_decay'] is not None else (lr, 0)
    l2                         = config['l2']
    mini_batch_percentage      = config['mini_batch_percentage']
    max_unlucky_epochs         = config['max_unlucky_epochs']
    max_epochs                 = config['max_epochs']
    min_train_error            = config['min_train_error'] if 'min_train_error' in config and config['min_train_error'] is not None else 0.0

    # Initialize the module-specific values that need to be used in each epoch.

    if config["optimizer"] == 'SGD' or config['optimizer'] == 'SGDN':
        # Momentum specific time-persistent values
        momentum     = config['momentum']
        prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
        prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]
    elif config["optimizer"] == 'adam':
        # Adam specific time-persistent values
        decay_rate_1 = config['adam_decay_rate_1']
        decay_rate_2 = config['adam_decay_rate_2']
        delta = 1e-8
        prev_first_delta_W  = [np.zeros(layer["W"].shape) for layer in model["layers"]] # s
        prev_first_delta_b  = [np.zeros(layer["b"].shape) for layer in model["layers"]]
        prev_second_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]] # r
        prev_second_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]

    # Split the datasets into inputs and outputs, if provided

    train_inputs    = training  [:, :model["in_dimension"]]
    train_target    = training  [:,  model["in_dimension"]:]
    val_inputs      = validation[:, :model["in_dimension"]]  if validation is not None else None
    val_target      = validation[:,  model["in_dimension"]:] if validation is not None else None
    watch_inputs    = watching  [:, :model["in_dimension"]]  if watching   is not None else None
    watch_target    = watching  [:,  model["in_dimension"]:] if watching   is not None else None

    # Initialize the metrics collected during training.

    train_errors, train_accuracies, val_errors, val_accuracies, watch_errors, watch_accuracies = [],[],[],[],[],[]
    metric_train_errors,metric_val_errors,metric_watch_errors = [], [], []

    # Initialize early stopping information

    unlucky_epochs = 0
    best_epoch = 0
    best_val_error = inf

    batch_size = int(mini_batch_percentage * training.shape[0])

    # For each epoch until the max_epoch is reached or early stopping/patience decides to stop the run:

    for epoch in range(max_epochs):

        # On each epoch, shuffle the training data

        dataset = np.random.permutation(training)

        # For each minibatch, apply the specific weights update procedure

        for i in range(0, ceil(dataset.shape[0] / batch_size)):

            # Construct the minibatch from the dataset

            mini_batch = dataset[i * batch_size:(i * batch_size) + batch_size][:]

            # Compute the gradient

            nabla_W, nabla_b = compute_gradient(model, mini_batch, loss_function)

            # Finally, pass it to the optimizer specified

            if config["optimizer"] == 'SGD':
                gradient_descent_step(model=model, epoch=epoch, prev_delta_W=prev_delta_W, prev_delta_b=prev_delta_b, nabla_W=nabla_W, nabla_b=nabla_b, lr_initial=lr, lr_final=lr_final, lr_final_epoch=lr_final_epoch, l2=l2, momentum=momentum)
            elif config["optimizer"] == 'SGDN':
                SGDN_step(model=model, mini_batch=mini_batch, loss_function=loss_function, epoch=epoch, prev_delta_W=prev_delta_W, prev_delta_b=prev_delta_b, lr_initial=lr, lr_final=lr_final, lr_final_epoch=lr_final_epoch, l2=l2, momentum=momentum)
            elif config["optimizer"] == 'adam':
                adam_step(model, epoch + 1, nabla_W, nabla_b, prev_first_delta_W, prev_first_delta_b, prev_second_delta_W, prev_second_delta_b, lr, l2, decay_rate_1, decay_rate_2, delta)

        # At the end of the training epoch, predict the model's output
        # on each dataset, training, validation, and watch if provided.
        # The learning curves refer to the error used for training the network.

        train_outputs          = predict(model, train_inputs)
        current_train_error    = loss_function.eval(train_outputs, train_target)
        current_train_accuracy = accuracy(train_outputs, train_target, target_domain) if target_domain is not None else inf

        val_outputs            = predict(model, val_inputs)                       if validation is not None else None
        current_val_error      = loss_function.eval(val_outputs, val_target)      if validation is not None else inf
        current_val_accuracy   = accuracy(val_outputs, val_target, target_domain) if validation is not None and target_domain is not None else inf

        watch_outputs          = predict(model, watch_inputs)                         if watching is not None else None
        current_watch_error    = loss_function.eval(watch_outputs, watch_target)      if watching is not None else inf
        current_watch_accuracy = accuracy(watch_outputs, watch_target, target_domain) if watching is not None and target_domain is not None else inf

        watch_errors.append(current_watch_error)
        watch_accuracies.append(current_watch_accuracy)

        train_errors.append(current_train_error)
        train_accuracies.append(current_train_accuracy)

        val_errors.append(current_val_error)
        val_accuracies.append(current_val_accuracy)

        # Compute the error with an additional metric different from the loss_function, if required

        if config['additional_metric'] is not None:
            metric_train_error    = mean_euclidean_error(train_outputs, train_target)
            metric_val_error      = mean_euclidean_error(val_outputs,   val_target)   if validation is not None else -1
            metric_watch_error    = mean_euclidean_error(watch_outputs, watch_target) if watching   is not None else -1

            metric_train_errors.append(metric_train_error)
            metric_val_errors.append(metric_val_error)
            metric_watch_errors.append(metric_watch_error)

        # Optional early stopping logic where a minimum error is provided

        if current_train_error <= min_train_error:
            break

        # Early stopping with cross validation logic: keep training
        # until the validation stops updating its minimum for a given
        # number of epochs unlucky_epochs.

        if validation is not None:
            if current_val_error <= best_val_error:
                best_val_error = current_val_error
                best_epoch = epoch
                unlucky_epochs = 0
            elif unlucky_epochs == max_unlucky_epochs:
                # We reached the end of the patience, end the training
                break
            else:
                unlucky_epochs += 1

    # Return the standard gradient descent information.

    return {'best_val_error':   best_val_error,
            'best_epoch':       best_epoch,
            'train_errors':     train_errors,
            'train_accuracies': train_accuracies,
            'val_errors':       val_errors,
            'val_accuracies':   val_accuracies,
            'watch_errors':     watch_errors,
            'watch_accuracies': watch_accuracies,
            'metric_train_errors': metric_train_errors,
            'metric_val_errors':   metric_val_errors,
            'metric_watch_errors': metric_watch_errors}

def gradient_descent_step(model, epoch, prev_delta_W, prev_delta_b, nabla_W, nabla_b, lr_initial, lr_final, lr_final_epoch, l2, momentum):
    """
    Specific weights update procedure for the Classical momentum.
    This is simply called as a modular replacement for the standard Gradient Descent procedure.
    :param model: model configuration object
    :epoch: current epoch
    :prev_delta_W: previous delta matrix (weights)
    :prev_delta_b: previous delta matrix (biases)
    :nabla_W: current network gradient (weights)
    :nabla_b: current network gradient (biases)
    :lr_initial: initial learning rate for learning rate decay
    :lr_final: target learning rate for learning rate decay
    :lr_final_epoch: terminal epoch for learning rate decay
    :l2: L2 regularization coefficient
    :momentum: Classical Momentum coefficient
    """

    # Linear rate decay:

    if epoch >= lr_final_epoch:
        alpha = 1.0 # Simply stop the decay
    else:
        alpha = epoch / lr_final_epoch # Take the percentage

    # Calculate the current learning rate by taking the linear interpolation.

    lr = (1.0 - alpha) * lr_initial + alpha * lr_final

    # Update the weights directly with the given classical momentum and learning rate

    for i in range(len(nabla_W)):
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr * nabla_W[i]
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr * nabla_b[i]

        model["layers"][i]["W"] += prev_delta_W[i] - 2*l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += prev_delta_b[i]
