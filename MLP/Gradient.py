from MLP.Layers import net
import numpy as np

def compute_gradient(model, mini_batch, error_function):
    """
    Compute the gradient of the current network given mini_batch and the loss function.
    :param model: network model
    :param mini_batch: (nparray) dataset to evaluate
    :param error_function: output error function for which the gradient is calculated
    :return nabla_W, nabla_b
    """

    # Separate the input and output data from the minibatch

    x = mini_batch[:,:model["in_dimension"]]
    y = mini_batch[:,model["in_dimension"]:]

    deltas = []
    activations = [x]
    activations_der = []

    # Feed-forward the input from the first layer to the output, collecting the results

    for layer in model["layers"]:

        # Initialize the deltas for each layer
        
        deltas.append(np.zeros(layer["W"].shape))

        # Apply feed-forward on the layers, saving the outputs
        
        net_x = net(layer, x)
        x = layer["activation_func"](net_x)

        # Append all the activations and their delta to the array

        activations_der.append(layer["activation_func_derivative"](net_x))
        activations.append(x)

    # Compute the deltas for the output layer

    deltas[-1] = error_function.gradient(activations[-1], y) * activations_der[-1]

    # Compute the deltas for the hidden layers

    for i in reversed(range(len(model["layers"])-1)):
        deltas[i] = np.dot(deltas[i+1], model["layers"][i+1]['W']) * activations_der[i]

    # Finally recollect the deltas by multiplying them by the activations and divide by the minibatch size 

    batch_size = x.shape[0]
    nabla_W = [d.T.dot(activations[i])/batch_size for i, d in enumerate(deltas)]
    nabla_b = [d.sum(axis=0)/batch_size for d in deltas]

    return (nabla_W, nabla_b)