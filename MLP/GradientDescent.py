from math import ceil
import numpy as np
from MLP.LossFunctions import loss_function_from_name, accuracy, MSE, CrossEntropy
from MLP.Layers import forward, net
from MLP.Network import model_predict_batch 
from MLP.Adam import adam_step
from math import inf

def print_epoch_stats(loss_function, model, t, train_error, val_error, watch_error):
    msg1 = f'| Train = {train_error:>18}\t'    if train_error else ''
    msg2 = f'| Validation = {val_error:>18}\t' if val_error else ''
    msg3 = f'| Watch = {watch_error:>18}\t'    if watch_error else ''
    print(f'Epoch {t+1} | ({loss_function.name}) ' + msg1 + msg2 + msg3)

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

    target_domain         = config['target_domain']
    loss_function         = loss_function_from_name(config['loss_function_name'])
    lr                    = config['lr']
    l2                    = config['l2']
    mini_batch_percentage = config['mini_batch_percentage']
    max_unlucky_epochs    = config['max_unlucky_epochs']
    max_epochs            = config['max_epochs']
    print_stats           = config['print_stats']

    if config["optimizer"] == 'SGD':
        momentum              = config['momentum']
        # Initialize to 0 the prev delta (used for momentum)
        prev_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]]
        prev_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]
    elif config["optimizer"] == 'adam':
        decay_rate_1 = config['decay_rate_1']
        decay_rate_2 = config['decay_rate_2']
        delta = 1e-8 # used for numerical stability
        prev_first_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]] # s
        prev_first_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]
        prev_second_delta_W = [np.zeros(layer["W"].shape) for layer in model["layers"]] # r
        prev_second_delta_b = [np.zeros(layer["b"].shape) for layer in model["layers"]]
    
    train_inputs    = training  [:, :model["in_dimension"]]
    train_target    = training  [:,  model["in_dimension"]:]
    val_inputs      = validation[:, :model["in_dimension"]]  if validation is not None else None
    val_target      = validation[:,  model["in_dimension"]:] if validation is not None else None
    watch_inputs    = watching  [:, :model["in_dimension"]]  if watching   is not None else None
    watch_target    = watching  [:,  model["in_dimension"]:] if watching    is not None else None

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

    for t in range(max_epochs):
        # Shuffle the training data
        # TODO: Sample the mini-batches correctly? This sampling is not i.i.d.
        dataset = np.random.permutation(training)
        # For each minibatch apply gradient descent
        for i in range(0, ceil(dataset.shape[0] / batch_size)):
            mini_batch = dataset[i * batch_size:(i * batch_size) + batch_size][:]
            
            nabla_W, nabla_b = compute_gradient(model, mini_batch, loss_function)

            if config["optimizer"] == 'SGD':
                gradient_descent_step(model, prev_delta_W, prev_delta_b, nabla_W, nabla_b, lr, l2, momentum)
            elif config["optimizer"] == 'adam':
                adam_step(nabla_W, nabla_b,
                          prev_first_delta_W, prev_first_delta_b, # s
                          prev_second_delta_W, prev_second_delta_b, # r
                          lr,
                          l2,
                          t + 1,
                          decay_rate_1,
                          decay_rate_2,
                          delta,
                          model)
            weights_norms.append(compute_weights_norm(model))
            gradient_norms.append(compute_gradient_norm(nabla_W, nabla_b))

        train_outputs          = model_predict_batch(model, train_inputs)
        current_train_error    = loss_function.eval(train_outputs, train_target)
        current_train_accuracy =           accuracy(train_outputs, train_target, target_domain)

        watch_outputs          = model_predict_batch(model, watch_inputs)                       if watching is not None else None
        current_watch_error    = loss_function.eval(watch_outputs, watch_target)                if watching is not None else inf
        current_watch_accuracy =           accuracy(watch_outputs, watch_target, target_domain) if watching is not None else inf

        val_outputs            = model_predict_batch(model, val_inputs)                     if validation is not None else None
        current_val_error      = loss_function.eval(val_outputs, val_target)                if validation is not None else inf
        current_val_accuracy   =           accuracy(val_outputs, val_target, target_domain) if validation is not None else inf

        train_errors.append(current_train_error)
        train_accuracies.append(current_train_accuracy)

        watch_errors.append(current_watch_error)
        watch_accuracies.append(current_watch_accuracy)

        val_errors.append(current_val_error)
        val_accuracies.append(current_val_accuracy)

        # Early stopping logic
        if validation is not None:
            if current_val_error <= best_val_error:
                best_val_error = current_val_error
                best_epoch = t
                unlucky_epochs = 0
            elif unlucky_epochs == max_unlucky_epochs:
                break
            else:
                unlucky_epochs += 1

        if print_stats:
            print_epoch_stats(loss_function, model, t, current_train_error, current_val_error, current_watch_error)

    return {'val_error':        best_val_error,
            'best_epoch':       best_epoch,
            'train_errors':     train_errors,
            'train_accuracies': train_accuracies,
            'val_errors':       val_errors,
            'val_accuracies':   val_accuracies,
            'watch_errors':     watch_errors,
            'watch_accuracies': watch_accuracies,
            'weights_norms':    weights_norms,
            'gradient_norms':   gradient_norms}

def gradient_descent_step(model, prev_delta_W, prev_delta_b, nabla_W, nabla_b, lr, l2, momentum):
    for i in range(len(nabla_W)):
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr * nabla_W[i]
        prev_delta_b[i] = momentum * prev_delta_b[i] - lr * nabla_b[i]

        model["layers"][i]["W"] += prev_delta_W[i] - l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += prev_delta_b[i] - l2 * model["layers"][i]["b"]

def compute_gradient(model, mini_batch, loss_function):
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

    return [nw/batch_size for nw in nabla_W], [nb/batch_size for nb in nabla_b]

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