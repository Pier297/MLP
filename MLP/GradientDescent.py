from math import ceil
import numpy as np
from MLP.LossFunctions import loss_function_from_name, accuracy, MSE, CrossEntropy
from MLP.Layers import forward, net
from MLP.Network import predict
from MLP.Adam import adam_step
from math import inf

def print_epoch_stats(loss_function, model, epoch, train_error, val_error, watch_error):
    msg1 = f'| Train = {train_error:<24}'    if train_error else ''
    msg2 = f'| Validation = {val_error:<24}' if val_error else ''
    msg3 = f'| Watch = {watch_error:<24}'    if watch_error else ''
    print(f'Epoch {epoch+1} | ({loss_function.name}) ' + msg1 + msg2 + msg3)

def gradient_descent(model, training, validation=None, config={}, watching=None):
    def compute_weights_norm(model):
        norm = 0.0
        for layer in model["layers"]:
            norm += np.linalg.norm(layer["W"]) + np.linalg.norm(layer["b"])
        return norm

    def compute_gradient_norm(nabla_W, nabla_b):
        norm = 0.0
        for nw, nb in zip(nabla_W, nabla_b):
            norm += np.linalg.norm(nw) + np.linalg.norm(nb)
        return norm

    target_domain              = config['target_domain']
    loss_function              = loss_function_from_name(config['loss_function_name'])
    lr                         = config['lr']
    (lr_final, lr_final_epoch) = config['lr_decay'] if config['lr_decay'] is not None else (lr, 0)
    l2                         = config['l2']
    mini_batch_percentage      = config['mini_batch_percentage']
    max_unlucky_epochs         = config['max_unlucky_epochs']
    max_epochs                 = config['max_epochs']
    print_stats                = config['print_stats']
    min_train_error            = config['min_train_error'] if 'min_train_error' in config and config['min_train_error'] is not None else 0.0

    if config["optimizer"] == 'SGD':
        momentum     = config['momentum']
        # Initialize to 0 the prev delta (used for momentum)
        prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
        prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]
    elif config["optimizer"] == 'adam':
        decay_rate_1 = config['adam_decay_rate_1']
        decay_rate_2 = config['adam_decay_rate_2']
        delta = 1e-8 # used for numerical stability
        prev_first_delta_W  = [np.zeros(layer["W"].shape) for layer in model["layers"]] # s
        prev_first_delta_b  = [np.zeros(layer["b"].shape) for layer in model["layers"]]
        prev_second_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]] # r
        prev_second_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]

    train_inputs    = training  [:, :model["in_dimension"]]
    train_target    = training  [:,  model["in_dimension"]:]
    val_inputs      = validation[:, :model["in_dimension"]]  if validation is not None else None
    val_target      = validation[:,  model["in_dimension"]:] if validation is not None else None
    watch_inputs    = watching  [:, :model["in_dimension"]]  if watching   is not None else None
    watch_target    = watching  [:,  model["in_dimension"]:] if watching   is not None else None

    train_errors        = []
    train_accuracies    = []
    val_errors          = []
    val_accuracies      = []
    watch_errors        = []
    watch_accuracies    = []

    weights_norms  = []
    gradient_norms = []

    unlucky_epochs = 0
    best_epoch = 0
    best_val_error = inf

    batch_size = int(mini_batch_percentage * training.shape[0])

    for epoch in range(max_epochs):
        # Shuffle the training data
        # TODO: Sample the mini-batches correctly? This sampling is not i.i.d.
        dataset = np.random.permutation(training)
        # For each minibatch apply gradient descent
        for i in range(0, ceil(dataset.shape[0] / batch_size)):
            mini_batch = dataset[i * batch_size:(i * batch_size) + batch_size][:]

            nabla_W, nabla_b = compute_gradient(model, mini_batch, loss_function)

            if config["optimizer"] == 'SGD':
                gradient_descent_step(model, epoch, prev_delta_W, prev_delta_b, nabla_W, nabla_b, lr, lr_final, lr_final_epoch, l2, momentum)
            elif config["optimizer"] == 'adam':
                adam_step(model,
                          epoch + 1,
                          nabla_W, nabla_b,
                          prev_first_delta_W, prev_first_delta_b, # s
                          prev_second_delta_W, prev_second_delta_b, # r
                          lr,
                          l2,
                          decay_rate_1,
                          decay_rate_2,
                          delta)
            weights_norms.append(compute_weights_norm(model))
            gradient_norms.append(compute_gradient_norm(nabla_W, nabla_b))

        train_outputs          = predict(model, train_inputs)
        current_train_error    = loss_function.eval(train_outputs, train_target)
        current_train_accuracy = accuracy(train_outputs, train_target, target_domain)

        val_outputs            = predict(model, val_inputs)                       if validation is not None else None
        current_val_error      = loss_function.eval(val_outputs, val_target)      if validation is not None else inf
        current_val_accuracy   = accuracy(val_outputs, val_target, target_domain) if validation is not None else inf

        watch_outputs          = predict(model, watch_inputs)                         if watching is not None else None
        current_watch_error    = loss_function.eval(watch_outputs, watch_target)      if watching is not None else inf
        current_watch_accuracy = accuracy(watch_outputs, watch_target, target_domain) if watching is not None else inf

        train_errors.append(current_train_error)
        train_accuracies.append(current_train_accuracy)

        watch_errors.append(current_watch_error)
        watch_accuracies.append(current_watch_accuracy)

        val_errors.append(current_val_error)
        val_accuracies.append(current_val_accuracy)

        # Early stopping with max accuracy logic
        if current_train_error <= min_train_error:
            break

        # Early stopping with cross validation logic
        if validation is not None:
            if current_val_error <= best_val_error:
                best_val_error = current_val_error
                best_epoch = epoch
                unlucky_epochs = 0
            elif unlucky_epochs == max_unlucky_epochs:
                break
            else:
                unlucky_epochs += 1

        if print_stats:
            print_epoch_stats(loss_function, model, epoch, current_train_error, current_val_error, current_watch_error)

    return {'best_val_error':   best_val_error,
            'best_epoch':       best_epoch,
            'train_errors':     train_errors,
            'train_accuracies': train_accuracies,
            'val_errors':       val_errors,
            'val_accuracies':   val_accuracies,
            'watch_errors':     watch_errors,
            'watch_accuracies': watch_accuracies,
            'weights_norms':    weights_norms,
            'gradient_norms':   gradient_norms}

def gradient_descent_step(model, epoch, prev_delta_W, prev_delta_b, nabla_W, nabla_b, lr_initial, lr_final, lr_final_epoch, l2, momentum):
    if epoch >= lr_final_epoch:
        alpha = 1.0
    else:
        alpha = epoch / lr_final_epoch

    lr = (1.0 - alpha) * lr_initial + alpha * lr_final

    for i in range(len(nabla_W)):

        prev_delta_W[i] = momentum * prev_delta_W[i] - lr * nabla_W[i]
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr * nabla_b[i]

        model["layers"][i]["W"] += prev_delta_W[i] - l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += prev_delta_b[i] - l2 * model["layers"][i]["b"]

def compute_gradient(model, mini_batch, loss_function):
    x = mini_batch[:,:model["in_dimension"]]
    y = mini_batch[:,model["in_dimension"]:]

    # Forward computation
    deltas = []
    activations = [x]
    activations_der = []

    for layer in model["layers"]:
        deltas.append(np.zeros(layer["W"].shape))
        net_x = net(layer, x)
        x = layer["activation_func"](net_x)
        activations_der.append(layer["activation_func_derivative"](net_x))
        activations.append(x)

    # Output layer
    deltas[-1] = loss_function.gradient(activations[-1], y) * activations_der[-1]

    # Hidden layers
    for i in reversed(range(len(model["layers"])-1)):
        deltas[i] = np.dot(deltas[i+1], model["layers"][i+1]['W']) * activations_der[i]

    batch_size = x.shape[0]

    nabla_W = [d.T.dot(activations[i])/batch_size for i, d in enumerate(deltas)]
    nabla_b = [d.sum(axis=0)/batch_size for d in deltas]

    return (nabla_W, nabla_b)
