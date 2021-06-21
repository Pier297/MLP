import numpy as np

def adam_step(
    nabla_W,
    nabla_b,
    prev_first_delta_W,
    prev_first_delta_b,
    prev_second_delta_W,
    prev_second_delta_b,
    lr,
    l2,
    t,
    decay_rate_1,
    decay_rate_2,
    delta,
    model):

    prev_first_delta_W = [decay_rate_1 * prev_first_delta_W[i] + (1 - decay_rate_1) * nabla_W[i] for i in range(len(prev_first_delta_W))]
    prev_first_delta_b = [decay_rate_1 * prev_first_delta_b[i] + (1 - decay_rate_1) * nabla_b[i] for i in range(len(prev_first_delta_b))]
    prev_second_delta_W = [decay_rate_2 * prev_second_delta_W[i] + (1 - decay_rate_2) * (nabla_W[i] * nabla_W[i]) for i in range(len(prev_second_delta_W))]
    prev_second_delta_b = [decay_rate_2 * prev_second_delta_b[i] + (1 - decay_rate_2) * (nabla_b[i] * nabla_b[i]) for i in range(len(prev_second_delta_b))]
    
    correct_first_delta_W = [s / (1 - decay_rate_1**t) for s in prev_first_delta_W]
    correct_first_delta_b = [s / (1 - decay_rate_1**t) for s in prev_first_delta_b]
    correct_second_delta_W = [r / (1 - decay_rate_2**t) for r in prev_second_delta_W]
    correct_second_delta_b = [r / (1 - decay_rate_2**t) for r in prev_second_delta_b]

    # Update parameters
    for i in range(len(nabla_W)):
        delta_W = -lr * correct_first_delta_W[i] / (np.sqrt(correct_second_delta_W[i]) + delta)
        delta_b = -lr * correct_first_delta_b[i] / (np.sqrt(correct_second_delta_b[i]) + delta)

        model["layers"][i]["W"] += delta_W - l2 * model["layers"][i]["W"]
        model["layers"][i]["b"] += delta_b - l2 * model["layers"][i]["b"]
