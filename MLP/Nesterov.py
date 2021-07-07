from MLP.Gradient import compute_gradient
import numpy as np

# Implementation of Nesterov Momentum.
# This function implements a single weights update step in GradientDescent.py.
def nag_step(model={}, mini_batch=[], loss_function=None, epoch=0, prev_delta_W=[], prev_delta_b=[], lr_initial=0, lr_final=0, lr_final_epoch=0, l2=0, momentum=0):
    # Linear rate decay:
    if epoch >= lr_final_epoch:
        alpha = 1.0 # Simply stop the decay
    else:
        alpha = epoch / lr_final_epoch # Take the percentage

    # Calculate the current learning rate by taking the linear interpolation.
    lr = (1.0 - alpha) * lr_initial + alpha * lr_final

    # Initialize the old layer weights
    old_layers_W = []
    old_layers_b = []
    for i in range(len(model["layers"])):
        old_layers_W.append(np.array(model["layers"][i]['W']))
        old_layers_b.append(np.array(model["layers"][i]['b']))

    # Apply interim update for each layer
    for i in range(len(model["layers"])):
        model["layers"][i]['W'] = model["layers"][i]['W'] + momentum * prev_delta_W[i]
        if model["layers"][i]['use_bias']:
            model["layers"][i]['b'] = model["layers"][i]['b'] + momentum * prev_delta_b[i]

    # Compute the gradient
    nabla_W, nabla_b = compute_gradient(model, mini_batch, loss_function)

    # Compute velocity update and apply update
    for i in range(len(nabla_W)):
        prev_delta_W[i] = momentum * prev_delta_W[i] - lr * nabla_W[i]
        model["layers"][i]['W'] = old_layers_W[i] + prev_delta_W[i] - 2*l2 * model["layers"][i]["W"]
        if model["layers"][i]['use_bias']:
            prev_delta_b[i] = momentum * prev_delta_b[i] - lr * nabla_b[i]
            model["layers"][i]['b'] = old_layers_b[i] + prev_delta_b[i]
