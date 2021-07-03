from MLP.Layers import net
import numpy as np

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